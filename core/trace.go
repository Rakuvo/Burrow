package core

import (
	"encoding/json"
	"io"
	"sync"
	"time"
)

// Event is the trace record for one attempt of one step.
type Event struct {
	// Agent is the caller's name for the agent that made the call (the X-Rakuvo-Agent header), or "".
	Agent    string
	RunID    string
	Attempt  int
	Provider string
	Model    string
	Latency  time.Duration
	Usage    Usage
	// AuxUsage is token use by layers on other models (e.g. a judge) in this attempt.
	AuxUsage Usage
	Err      string
	Retried  bool
	Notes    []Note
}

// Tracer receives one Event per attempt. Implementations must be safe for
// concurrent use.
type Tracer interface {
	Emit(Event)
}

// MemoryTracer keeps events in memory. It backs tests.
type MemoryTracer struct {
	mu     sync.Mutex
	events []Event
}

func (t *MemoryTracer) Emit(e Event) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.events = append(t.events, e)
}

// Events returns a copy of everything emitted so far.
func (t *MemoryTracer) Events() []Event {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]Event(nil), t.events...)
}

// JSONLTracer writes one JSON object per attempt to w. It records tokens,
// timing, retries and layer notes, and deliberately nothing from prompts or
// responses, so a trace file is safe to keep and share.
type JSONLTracer struct {
	mu  sync.Mutex
	enc *json.Encoder
}

func NewJSONLTracer(w io.Writer) *JSONLTracer { return &JSONLTracer{enc: json.NewEncoder(w)} }

func (t *JSONLTracer) Emit(e Event) {
	type note struct {
		Layer   string `json:"layer"`
		Message string `json:"message"`
	}
	rec := struct {
		Time         string  `json:"time"`
		RunID        string  `json:"run_id"`
		Attempt      int     `json:"attempt"`
		Provider     string  `json:"provider"`
		Model        string  `json:"model"`
		LatencyMs    float64 `json:"latency_ms"`
		InputTokens  int     `json:"input_tokens"`
		OutputTokens int     `json:"output_tokens"`
		CachedTokens int     `json:"cached_tokens"`
		AuxIn        int     `json:"aux_input_tokens,omitempty"`
		AuxOut       int     `json:"aux_output_tokens,omitempty"`
		Error        string  `json:"error,omitempty"`
		Retried      bool    `json:"retried"`
		Notes        []note  `json:"notes,omitempty"`
	}{
		Time: time.Now().UTC().Format(time.RFC3339Nano), RunID: e.RunID, Attempt: e.Attempt,
		Provider: e.Provider, Model: e.Model, LatencyMs: float64(e.Latency) / float64(time.Millisecond),
		InputTokens: e.Usage.InputTokens, OutputTokens: e.Usage.OutputTokens, CachedTokens: e.Usage.CachedTokens,
		AuxIn: e.AuxUsage.InputTokens, AuxOut: e.AuxUsage.OutputTokens,
		Error: e.Err, Retried: e.Retried,
	}
	for _, n := range e.Notes {
		rec.Notes = append(rec.Notes, note{n.Layer, n.Message})
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	_ = t.enc.Encode(rec) // tracing must never fail a request
}

type nopTracer struct{}

func (nopTracer) Emit(Event) {}
