// Package effort is the reasoning-effort control. Reasoning models spend hidden
// "thinking" tokens that are billed as output, and on tool-calling work that was about
// 60% of the bill for gpt-5-nano. Setting a lower effort is the largest cost lever the
// harness has; this layer sets it per call, and can use a higher effort once Model
// Policy has escalated a call, so easy calls stay cheap and failed ones get more thought.
//
// The value is sent as chat completions' reasoning_effort. Which values a model accepts
// differs ("minimal" for gpt-5-nano and gpt-5-mini, "none" for newer models, and the
// o-series cannot turn reasoning off), so a value the model rejects comes back as a
// provider error; with Model Policy in front, that error escalates the call.
package effort

import (
	"context"
	"encoding/json"

	"github.com/rakuvo/burrow/core"
)

// StateEscalated mirrors modelpolicy.StateEscalated, so the two layers cooperate
// without importing each other.
const StateEscalated = "modelpolicy.escalated"

// Config selects the effort. The zero value changes nothing.
type Config struct {
	// Effort applies to every call: none, minimal, low, medium or high.
	Effort string
	// EscalatedEffort applies instead once Model Policy has moved the call to the strong
	// model. Empty means the same as Effort.
	EscalatedEffort string
	// Force overrides an effort the caller set itself; by default the caller's wins.
	Force bool
}

// Valid lists the accepted values.
func Valid(v string) bool {
	switch v {
	case "", "none", "minimal", "low", "medium", "high":
		return true
	}
	return false
}

// Layer implements core.Layer.
type Layer struct{ cfg Config }

func New(cfg Config) *Layer { return &Layer{cfg: cfg} }

func (*Layer) Name() string { return "effort" }

func (*Layer) After(context.Context, *core.Step) error { return nil }

func (l *Layer) Before(_ context.Context, s *core.Step) error {
	v := l.cfg.Effort
	if esc, _ := s.State[StateEscalated].(bool); esc && l.cfg.EscalatedEffort != "" {
		v = l.cfg.EscalatedEffort
	}
	if v == "" {
		return nil
	}
	if _, set := s.Request.Extra["reasoning_effort"]; set && !l.cfg.Force {
		return nil
	}
	if s.Request.Extra == nil {
		s.Request.Extra = map[string]json.RawMessage{}
	}
	raw, _ := json.Marshal(v)
	s.Request.Extra["reasoning_effort"] = raw
	s.Note(l.Name(), "reasoning effort set to "+v)
	return nil
}
