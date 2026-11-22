// Package modelpolicy is the Model Policy layer. It sends each request to a cheap model
// first and escalates to a strong model only when the cheap one demonstrably fails, so
// most calls cost cheap-model prices and the hard ones still get the strong model.
//
// "Fails" means something the harness can observe without knowing the right answer:
//
//   - another layer asked for a retry (Guard found an invalid or undecodable tool call,
//     Verify vetoed an unbelievable one, Loop saw a stuck pattern), or
//   - the provider returned an error or an empty response, or
//   - optionally, the cheap model answered in plain text although tools were offered.
//
// The retry then runs on the strong model, and stays on it for the rest of the call.
// A call that never fails never touches the strong model. Order matters: register this
// layer before the layers whose retries it should react to, because After hooks run in
// reverse registration order and this one has to see their decision.
//
// Measured, not assumed: a live test did not reproduce an earlier offline simulation of the
// savings. A reasoning model's hidden output tokens can make a "cheap" tier dearer than a
// non-reasoning "strong" model. Choose a cheap tier that is actually cheaper per task, and
// measure it on your own traffic.
package modelpolicy

import (
	"context"
	"fmt"

	"github.com/rakuvo/burrow/core"
)

// StateEscalated is true in Step.State once the call has moved to the strong model.
const StateEscalated = "modelpolicy.escalated"

// Config selects the two tiers. The zero value changes nothing.
type Config struct {
	// Cheap is tried first. Strong takes over when Cheap fails.
	Cheap, Strong string
	// EscalateOnNoCall also escalates when the cheap model answered in text although
	// tools were offered. Off by default: a text answer is often right (no tool applies).
	EscalateOnNoCall bool
	// DisableOnError stops provider errors (outages, rate limits) from escalating. By
	// default they do, which doubles as a fallback when the cheap model is unavailable.
	DisableOnError bool
}

// Layer implements core.Layer.
type Layer struct{ cfg Config }

func New(cfg Config) *Layer { return &Layer{cfg: cfg} }

func (*Layer) Name() string { return "modelpolicy" }

func (l *Layer) active() bool { return l.cfg.Cheap != "" && l.cfg.Strong != "" }

// applies reports whether this request is one the policy manages: the caller asked for
// one of the two tiers. Requests for any other model pass through untouched.
func (l *Layer) applies(model string) bool {
	return model == l.cfg.Cheap || model == l.cfg.Strong
}

func (l *Layer) Before(_ context.Context, s *core.Step) error {
	if !l.active() || !l.applies(s.Original.Model) {
		return nil
	}
	if esc, _ := s.State[StateEscalated].(bool); esc {
		s.Request.Model = l.cfg.Strong
		return nil
	}
	s.Request.Model = l.cfg.Cheap
	return nil
}

func (l *Layer) After(_ context.Context, s *core.Step) error {
	if !l.active() || !l.applies(s.Original.Model) {
		return nil
	}
	if esc, _ := s.State[StateEscalated].(bool); esc || s.Request.Model != l.cfg.Cheap {
		return nil // already on the strong tier, nothing left to escalate to
	}
	if s.Attempt >= s.MaxAttempts {
		return nil // no attempt left to run on the strong model
	}
	reason := ""
	switch {
	case s.Retry:
		reason = "another layer rejected the response"
	case s.Err != nil && !l.cfg.DisableOnError:
		reason = "provider error: " + s.Err.Error()
	case s.Err == nil && (s.Response == nil || emptyResponse(s.Response)):
		reason = "empty response"
	case l.cfg.EscalateOnNoCall && len(s.Request.Tools) > 0 && s.Response != nil && len(s.Response.Message.ToolCalls) == 0 && callExpected(s.Request):
		reason = "no tool call although tools were offered"
	}
	if reason == "" {
		return nil
	}
	s.State[StateEscalated] = true
	s.Retry = true
	s.Err = nil // the strong model's attempt replaces this one
	s.Note(l.Name(), fmt.Sprintf("escalating %s -> %s: %s", l.cfg.Cheap, l.cfg.Strong, reason))
	return nil
}

func emptyResponse(r *core.Response) bool {
	return r.Message.Content == "" && len(r.Message.ToolCalls) == 0
}

// callExpected reports whether the caller allows a tool call, i.e. did not set
// tool_choice to "none".
func callExpected(r core.Request) bool {
	raw, ok := r.Extra["tool_choice"]
	return !ok || string(raw) != `"none"`
}
