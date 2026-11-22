package api

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/rakuvo/burrow/layers/memory"
)

// MemorySync reports the gateway's memory graphs to the API so the dashboard can draw them. It is opt-in
// because facts are what users said, not metadata: they go only to the API the operator chose, and the
// original quotes are left out unless ReportQuotes is set. Each pass sends whole scopes by fact id, so it is
// idempotent, and a scope whose facts have not changed since the last pass is skipped.
type MemorySync struct {
	URL, Key     string
	Store        *memory.Store
	Interval     time.Duration // default 15s
	ReportQuotes bool
	Log          *slog.Logger
	Client       *http.Client

	last map[string][32]byte
}

// Run reports until ctx ends. It never fails a model call: an unreachable API is logged and retried next pass.
func (m *MemorySync) Run(ctx context.Context) {
	m.defaults()
	t := time.NewTicker(m.Interval)
	defer t.Stop()
	for {
		m.Pass(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (m *MemorySync) defaults() {
	if m.Interval <= 0 {
		m.Interval = 15 * time.Second
	}
	if m.Log == nil {
		m.Log = slog.Default()
	}
	if m.Client == nil {
		m.Client = &http.Client{Timeout: 15 * time.Second}
	}
	if m.last == nil {
		m.last = map[string][32]byte{}
	}
}

// Pass sends every scope that changed since the last pass and reports how many scopes it sent.
func (m *MemorySync) Pass(ctx context.Context) int {
	m.defaults()
	sent := 0
	for _, scope := range m.Store.Scopes() {
		facts := m.facts(scope)
		if len(facts) == 0 {
			continue
		}
		body, err := json.Marshal(map[string]any{"scope": scope, "facts": facts})
		if err != nil {
			continue
		}
		sum := sha256.Sum256(body)
		if m.last[scope] == sum {
			continue
		}
		if err := m.post(ctx, facts, scope); err != nil {
			m.Log.Warn("memory report failed; will retry", "err", err)
			continue
		}
		m.last[scope] = sum
		sent++
	}
	return sent
}

func (m *MemorySync) facts(scope string) []MemoryFact {
	return MemoryFactsOf(m.Store, scope, m.ReportQuotes)
}

// MemoryFactsOf converts one scope of a memory store to the API's records of its facts. The user's original
// wording is included only when quotes is true.
func MemoryFactsOf(st *memory.Store, scope string, quotes bool) []MemoryFact {
	var out []MemoryFact
	for _, f := range st.Dump(scope) {
		mf := MemoryFact{ID: f.ID, Subject: f.Subject, Predicate: f.Predicate, Object: f.Object, Text: f.Text,
			Single: f.Single, Created: f.Created, Superseded: f.Superseded, Seen: f.Seen}
		if quotes {
			mf.Quote = f.Quote
		}
		out = append(out, mf)
	}
	return out
}

// post sends a scope in batches no larger than the API accepts.
func (m *MemorySync) post(ctx context.Context, facts []MemoryFact, scope string) error {
	url := strings.TrimRight(m.URL, "/") + "/v1/ingest/memory"
	for len(facts) > 0 {
		n := len(facts)
		if n > maxBatch {
			n = maxBatch
		}
		b, _ := json.Marshal(map[string]any{"scope": scope, "facts": facts[:n]})
		req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(b))
		req.Header.Set("Authorization", "Bearer "+m.Key)
		req.Header.Set("Content-Type", "application/json")
		resp, err := m.Client.Do(req)
		if err != nil {
			return err
		}
		resp.Body.Close()
		if resp.StatusCode/100 != 2 {
			return &httpError{resp.StatusCode}
		}
		facts = facts[n:]
	}
	return nil
}

type httpError struct{ code int }

func (e *httpError) Error() string { return "the API answered " + http.StatusText(e.code) }
