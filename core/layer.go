package core

import "context"

// Step is the state of one logical model call across all of its attempts.
// Layers read and mutate it; the engine owns its lifecycle.
type Step struct {
	// Original is the request exactly as the caller sent it. It is never modified.
	Original Request
	// Request is what will be sent to the provider on this attempt. The engine
	// resets it to a clone of Original before each attempt's Before hooks run.
	Request  Request
	Response *Response
	Err      error // provider error on this attempt, if any

	Attempt int
	// MaxAttempts is the engine's hard cap. A layer that finds a problem should
	// ask for a retry only while Attempt < MaxAttempts; on the last attempt it
	// must decide what to do with the bad response instead.
	MaxAttempts int
	// AuxUsage is what layers spent on other model calls during this attempt, such
	// as a judge. It is reset each attempt, reported in the trace, and kept apart
	// from the supervised model's own usage because it is usually a different, cheaper model.
	AuxUsage Usage
	// Retry asks the engine to run another attempt. Set it from After.
	Retry bool

	// State persists across attempts, so a layer can remember e.g. "escalate to a
	// bigger model next time".
	State map[string]any

	notes []Note
}

// Note records a layer decision for the trace.
type Note struct {
	Layer   string
	Message string
}

// Note adds a decision to this attempt's trace event.
func (s *Step) Note(layer, msg string) {
	s.notes = append(s.notes, Note{Layer: layer, Message: msg})
}

// Layer is one reliability layer. Before runs in registration order ahead of the
// provider call; After runs in reverse order once the provider has answered.
// Before may be called several times for one Step (once per attempt) and must
// not assume it has not seen this Step.
type Layer interface {
	Name() string
	Before(ctx context.Context, s *Step) error
	After(ctx context.Context, s *Step) error
}

// Base lets a layer implement only the hooks it needs.
type Base struct{}

func (Base) Before(context.Context, *Step) error { return nil }
func (Base) After(context.Context, *Step) error  { return nil }

// Keys in Step.State that more than one layer reads or writes. They live here so
// layers can cooperate without importing each other.
const (
	// StateNoToolPolicy: a layer sets it to true to make later attempts run
	// without the tool-use policy.
	StateNoToolPolicy = "core.no_tool_policy"
	// StateToolPolicyApplied is true on an attempt where the Prompt layer added
	// the tool-use policy. The Prompt layer rewrites it at the start of every attempt.
	StateToolPolicyApplied = "core.tool_policy_applied"
)
