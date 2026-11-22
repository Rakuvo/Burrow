// Package verify is the Verifier layer. A separate judge model checks one
// question, "does any offered tool apply to this request?", and the layer acts on
// the answer in two independent ways:
//
//   - Gate: the model answered in plain text although tools were offered. If the
//     judge says a tool applies, retry with a call required.
//   - Veto: the model made a call after the tool-use policy told it to prefer
//     calls. If the judge says no tool applies, retry without the policy so the
//     model behaves as it would have unprompted.
//
// Both are built to leave good behaviour alone. The judge sees only the request and
// the tool names, descriptions and required parameters; a judge that is down, slow or
// confused never blocks a response; and each action happens at most once per request.
package verify

import (
	"context"
	"encoding/json"

	"github.com/rakuvo/burrow/core"
)

const (
	keyRequire = "verify.require"
	keyVetoed  = "verify.vetoed"
)

// Config selects what the layer does. The zero value does nothing.
type Config struct {
	Gate bool // retry skipped tool calls with a call required
	Veto bool // retry policy-prompted calls without the policy when no tool applies

	// SingleRequestOnly consults the judge only when the conversation is one user request with
	// no tool results yet. Further into an agent session some tool nearly always applies, so the
	// judge's answer is known in advance and its retries only get in the way.
	SingleRequestOnly bool
}

// Layer implements core.Layer. It keeps no per-request state.
type Layer struct {
	judge Judger
	cfg   Config
}

func New(j Judger, cfg Config) *Layer { return &Layer{judge: j, cfg: cfg} }

func (*Layer) Name() string { return "verify" }

// Before makes a gate retry require a call.
func (*Layer) Before(_ context.Context, s *core.Step) error {
	if req, _ := s.State[keyRequire].(bool); req {
		if s.Request.Extra == nil {
			s.Request.Extra = map[string]json.RawMessage{}
		}
		s.Request.Extra["tool_choice"] = json.RawMessage(`"required"`)
	}
	return nil
}

func (l *Layer) After(ctx context.Context, s *core.Step) error {
	switch {
	case l.cfg.Gate && l.gateCandidate(s):
		return l.consult(ctx, s, func(applies bool) {
			if !applies {
				s.Note(l.Name(), "plain-text answer kept: the judge found no tool that applies")
				return
			}
			s.Note(l.Name(), "plain-text answer, but a tool applies: retrying with a call required")
			s.State[keyRequire] = true
			s.Retry = true
		})
	case l.cfg.Veto && l.vetoCandidate(s):
		return l.consult(ctx, s, func(applies bool) {
			if applies {
				s.Note(l.Name(), "call kept: the judge agrees a tool applies")
				return
			}
			s.Note(l.Name(), "call vetoed: the judge found no tool that applies; retrying without the tool-use policy")
			s.State[core.StateNoToolPolicy] = true
			s.State[keyVetoed] = true
			s.Retry = true
		})
	}
	return nil
}

// consult asks the judge and hands its verdict to act. A failing judge fails open.
func (l *Layer) consult(ctx context.Context, s *core.Step, act func(applies bool)) error {
	v, err := l.judge.Judge(ctx, s.Request)
	s.AuxUsage.InputTokens += v.Usage.InputTokens // judge tokens are real spend, even when the call failed
	s.AuxUsage.OutputTokens += v.Usage.OutputTokens
	if err != nil {
		s.Note(l.Name(), "judge unavailable; response kept ("+err.Error()+")")
		return nil
	}
	act(v.Applies)
	return nil
}

// common conditions: there is a response to supervise, tools were on offer, the
// caller did not dictate tool use, this is the turn that answers a user (not the
// reply to a tool result), and a retry is still possible.
func (l *Layer) supervisable(s *core.Step) bool {
	if s.Err != nil || s.Response == nil || len(s.Request.Tools) == 0 || s.Attempt >= s.MaxAttempts {
		return false
	}
	if msgs := s.Request.Messages; len(msgs) == 0 || msgs[len(msgs)-1].Role != core.RoleUser {
		return false // after a tool result, plain text is the finished answer, not a skipped call
	}
	if l.cfg.SingleRequestOnly && !singleRequest(s.Request.Messages) {
		return false
	}
	if raw, ok := s.Request.Extra["tool_choice"]; ok {
		var mode string
		if json.Unmarshal(raw, &mode) != nil || mode != "auto" {
			return false // the caller forbade calls ("none"), required them, or named one: not our decision
		}
	}
	return true
}

// singleRequest reports whether the conversation is one user message and nothing after it
// but system text: no earlier turns, no tool calls, no tool results.
func singleRequest(msgs []core.Message) bool {
	users := 0
	for _, m := range msgs {
		switch m.Role {
		case core.RoleUser:
			users++
		case core.RoleAssistant, core.RoleTool:
			return false
		}
	}
	return users == 1
}

// gateCandidate: a plain-text answer that might be a skipped tool call.
func (l *Layer) gateCandidate(s *core.Step) bool {
	return l.supervisable(s) && len(s.Response.Message.ToolCalls) == 0 && s.Response.FinishReason != "length"
}

// vetoCandidate: a call the tool-use policy may have talked the model into. Without
// the policy a call is the model's own choice, so it is left alone.
func (l *Layer) vetoCandidate(s *core.Step) bool {
	if !l.supervisable(s) || len(s.Response.Message.ToolCalls) == 0 {
		return false
	}
	if applied, _ := s.State[core.StateToolPolicyApplied].(bool); !applied {
		return false
	}
	vetoed, _ := s.State[keyVetoed].(bool)
	return !vetoed
}
