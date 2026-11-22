package api

import (
	"time"

	"github.com/rakuvo/burrow/core"
)

const (
	maxBatch      = 1000 // events per ingest call
	maxConfigSize = 64 << 10
)

// Note is a layer's decision recorded in an event.
type Note struct {
	Layer   string `json:"layer"`
	Message string `json:"message"`
}

// Event is one attempt of one model call, as the gateway's tracer reports it.
type Event struct {
	Agent     string    `json:"agent,omitempty"` // which of the customer's agents made the call (X-Rakuvo-Agent)
	RunID     string    `json:"run_id"`
	Attempt   int       `json:"attempt"`
	Time      time.Time `json:"time"`
	Provider  string    `json:"provider"`
	Model     string    `json:"model"`
	LatencyMs float64   `json:"latency_ms"`
	Input     int       `json:"input_tokens"`
	Output    int       `json:"output_tokens"`
	Cached    int       `json:"cached_tokens"`
	AuxIn     int       `json:"aux_input_tokens"`
	AuxOut    int       `json:"aux_output_tokens"`
	Error     string    `json:"error,omitempty"`
	Retried   bool      `json:"retried"`
	Notes     []Note    `json:"notes,omitempty"`
}

// MemoryFact is one edge of a user's memory graph: the same shape the gateway's
// memory layer stores, without the embedding vector.
type MemoryFact struct {
	ID         int       `json:"id"`
	Subject    string    `json:"subject"`
	Predicate  string    `json:"predicate"`
	Object     string    `json:"object"`
	Text       string    `json:"text"`
	Single     bool      `json:"single,omitempty"`
	Created    time.Time `json:"created"`
	Superseded time.Time `json:"superseded,omitempty"`
	Seen       int       `json:"seen,omitempty"`
	Quote      string    `json:"quote,omitempty"`
	Sources    []string  `json:"sources,omitempty"`
}

// FromCore converts a gateway trace event to the API's record of it.
func FromCore(e core.Event) Event {
	ev := Event{Agent: e.Agent, RunID: e.RunID, Attempt: e.Attempt, Time: time.Now().UTC(), Provider: e.Provider, Model: e.Model,
		LatencyMs: float64(e.Latency) / float64(time.Millisecond),
		Input:     e.Usage.InputTokens, Output: e.Usage.OutputTokens, Cached: e.Usage.CachedTokens,
		AuxIn: e.AuxUsage.InputTokens, AuxOut: e.AuxUsage.OutputTokens, Error: e.Err, Retried: e.Retried}
	for _, n := range e.Notes {
		ev.Notes = append(ev.Notes, Note{Layer: n.Layer, Message: n.Message})
	}
	return ev
}
