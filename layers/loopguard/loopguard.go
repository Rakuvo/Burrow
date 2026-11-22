// Package loopguard is the Loop layer. It stops an agent from burning tokens on a
// task it is not making progress on.
//
// The proxy never sees the agent's own state, but every request carries the whole
// conversation, so the layer reads the loop off the history: which tools were called,
// with what arguments, and what came back. Four detectors look for the signatures
// found in the agent-loop literature and in production agent SDKs:
//
//   - no-progress: the same call returned the same result again and again (the
//     clearest sign of a stuck agent, since nothing changed between tries);
//   - repeat: the same call many times whatever it returned (higher limits, because
//     polling is legitimate);
//   - cycle: a pattern of two or three different calls repeated (A,B,A,B,...);
//   - text: the assistant keeps saying nearly the same thing.
//
// Two tiers answer them. At the warning level the model is told it is stuck and shown
// what it has been doing, once, before its next try. At the stop level, if the model
// still asks for the looping call, the call is replaced by a plain message saying the
// task was stopped and why, which ends a well-behaved agent loop cleanly. A model can
// ignore a nudge but cannot talk its way past a halt. A hard cap on tool calls is
// available as well.
//
// The layer keeps no state between requests. It never halts a response that carries no
// tool call, and never halts when the caller forced tool use.
package loopguard

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/rakuvo/burrow/core"
)

// Action says what the layer does at the stop level.
type Action int

const (
	Halt     Action = iota // replace the looping call with a final message (default)
	WarnOnly               // only nudge and record; never alter a response
)

// Config selects what the layer does. Every zero threshold turns that detector off,
// so the zero value does nothing; Defaults gives the recommended settings.
type Config struct {
	// Window is how many recent tool calls the detectors look at (0 = all).
	Window int

	NoProgressWarn, NoProgressStop int // identical call with identical result, this many times
	RepeatWarn, RepeatStop         int // identical call, any result
	CycleWarn, CycleStop           int // repetitions of a 2- or 3-call cycle
	TextWarn, TextStop             int // consecutive near-identical assistant messages
	// TextSimilarityThreshold is the bigram overlap above which two messages count as
	// the same (0 means 0.85).
	TextSimilarityThreshold float64

	// ProgressExempt stops a result that reports work in progress ("running", "queued",
	// "pending") from counting as no-progress. Such a poll is waiting, not stuck; it is
	// held to PollWarn and PollStop identical calls instead, which are much higher than
	// RepeatWarn and RepeatStop. Results that mention an error never qualify.
	ProgressExempt     bool
	PollWarn, PollStop int

	// DryWarn and DryStop count consecutive empty or error results from one tool, whatever the
	// arguments: an agent trying variations of a request that cannot be satisfied.
	DryWarn, DryStop int

	// MaxToolCalls caps the tool calls in one conversation (0 = no cap). Calls in
	// history plus the new ones must not exceed it.
	MaxToolCalls int

	Action Action

	// AllowRepeat names tools that are meant to be called repeatedly, such as a
	// status poll or a wait. They are exempt from the repeat, no-progress and cycle
	// detectors, because a poll returns the same result until the job is done; the
	// MaxToolCalls cap is the backstop for a poll that never finishes.
	AllowRepeat []string
	// IgnoreArgs lists top-level argument names left out when comparing calls, for
	// values that change every time (timestamps, request IDs).
	IgnoreArgs []string
	// KeyArgs, per tool, lists the only arguments that decide whether two calls are
	// the same. It takes precedence over IgnoreArgs for that tool.
	KeyArgs map[string][]string
}

// TextSimilarity returns the effective similarity threshold.
func (c Config) TextSimilarity() float64 {
	if c.TextSimilarityThreshold > 0 {
		return c.TextSimilarityThreshold
	}
	return 0.85
}

// Defaults are the recommended settings: warn at three identical call-and-result
// pairs and stop at five; allow more room for repeats whose results differ.
func Defaults() Config {
	return Config{
		Window:         20,
		NoProgressWarn: 3, NoProgressStop: 5,
		RepeatWarn: 5, RepeatStop: 10,
		CycleWarn: 3, CycleStop: 5,
		TextWarn: 3, TextStop: 5,
		ProgressExempt: true, PollWarn: 20, PollStop: 40,
		DryWarn: 4, DryStop: 8,
	}
}

// Layer implements core.Layer. It keeps no per-request state.
type Layer struct{ cfg Config }

func New(cfg Config) *Layer { return &Layer{cfg: cfg} }

func (*Layer) Name() string { return "loop" }

const keyNudged = "loop.nudged"

// Before tells the model it is stuck when the history already shows a loop, or when
// an earlier attempt of this request found the model about to continue one.
func (l *Layer) Before(_ context.Context, s *core.Step) error {
	if len(s.Request.Tools) == 0 || toolsDisabled(s.Request) {
		return nil
	}
	findings := l.analyze(s.Request.Messages, nil)
	nudged, _ := s.State[keyNudged].(bool)
	w := worst(findings)
	if w == nil && !nudged {
		return nil
	}
	msg := nudgeText(findings)
	if msg == "" { // retried on a prediction: no finding in the history itself
		msg = genericNudge
	}
	s.State[keyNudged] = true // After must not ask for a second nudge on top of this one
	s.Request.Messages = append(s.Request.Messages, core.Message{Role: core.RoleSystem, Content: msg})
	s.Note(l.Name(), "told the model it appears stuck: "+firstDetail(findings))
	return nil
}

func (l *Layer) After(_ context.Context, s *core.Step) error {
	if s.Err != nil || s.Response == nil || len(s.Response.Message.ToolCalls) == 0 || len(s.Request.Tools) == 0 {
		return nil
	}
	cand := s.Response.Message.ToolCalls
	findings := l.analyze(s.Request.Messages, cand)
	if l.cfg.MaxToolCalls > 0 {
		if n := countCalls(s.Original.Messages) + len(cand); n > l.cfg.MaxToolCalls {
			findings = append(findings, Finding{Kind: "limit", Level: Stop, Count: n,
				Detail: fmt.Sprintf("this conversation has used %d tool calls, over the limit of %d", n, l.cfg.MaxToolCalls)})
		}
	}
	w := worst(findings)
	if w == nil || !forcedAuto(s.Request) {
		if w != nil {
			s.Note(l.Name(), "loop signs seen but the caller dictated tool use; left alone: "+w.Detail)
		}
		return nil
	}
	nudged, _ := s.State[keyNudged].(bool)
	switch {
	case w.Level == Stop && l.cfg.Action == Halt:
		s.Note(l.Name(), "stopped the loop: "+w.Detail)
		s.Response.Message.ToolCalls = nil
		s.Response.Message.Content = haltText(*w)
		s.Response.FinishReason = "stop"
	case w.Level >= Warn && !nudged && s.Attempt < s.MaxAttempts:
		s.State[keyNudged] = true
		s.Retry = true
		s.Note(l.Name(), "the model was about to continue a loop; retrying with a nudge: "+w.Detail)
	} // otherwise the model has been nudged already, and that is on record
	return nil
}

const genericNudge = "[Rakuvo Loop] You may be repeating yourself. Before calling a tool again, check whether the earlier calls already gave you what you need. If a call keeps returning the same result, repeating it will not change the outcome: try a different approach, change the arguments, or stop and tell the user what is blocking you."

func nudgeText(fs []Finding) string {
	w := worst(fs)
	if w == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString("[Rakuvo Loop] You appear to be stuck: ")
	b.WriteString(w.Detail)
	b.WriteString(". Repeating it will not change the outcome. Try a different approach, change the arguments, or stop and tell the user what is blocking you.")
	return b.String()
}

func haltText(f Finding) string {
	return fmt.Sprintf("Stopped by Rakuvo: the task looks stuck (%s). I stopped here instead of repeating it. Tell me how you would like to proceed, or change the request so a different approach is possible.", f.Detail)
}

func firstDetail(fs []Finding) string {
	if w := worst(fs); w != nil {
		return w.Detail
	}
	return "an earlier attempt found the model about to continue a loop"
}

func countCalls(msgs []core.Message) int {
	n := 0
	for _, m := range msgs {
		if m.Role == core.RoleAssistant {
			n += len(m.ToolCalls)
		}
	}
	return n
}

// toolsDisabled reports tool_choice "none".
func toolsDisabled(req core.Request) bool {
	raw, ok := req.Extra["tool_choice"]
	if !ok {
		return false
	}
	var mode string
	return json.Unmarshal(raw, &mode) == nil && mode == "none"
}

// forcedAuto reports that the caller left the choice of tool use to the model, so the
// layer may change it. A required or named tool_choice must not be answered with plain text.
func forcedAuto(req core.Request) bool {
	raw, ok := req.Extra["tool_choice"]
	if !ok {
		return true
	}
	var mode string
	return json.Unmarshal(raw, &mode) == nil && mode == "auto"
}
