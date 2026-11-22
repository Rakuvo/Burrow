package core_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/providers/mock"
)

// recorder logs hook order into a shared slice.
type recorder struct {
	core.Base
	name string
	log  *[]string
}

func (r recorder) Name() string { return r.name }
func (r recorder) Before(context.Context, *core.Step) error {
	*r.log = append(*r.log, r.name+".before")
	return nil
}
func (r recorder) After(context.Context, *core.Step) error {
	*r.log = append(*r.log, r.name+".after")
	return nil
}

func TestLayerOrder(t *testing.T) {
	var log []string
	e := &core.Engine{
		Provider: mock.New(mock.Text("hi")),
		Layers:   []core.Layer{recorder{name: "a", log: &log}, recorder{name: "b", log: &log}},
	}
	if _, err := e.Complete(context.Background(), core.Request{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	want := []string{"a.before", "b.before", "b.after", "a.after"}
	if !reflect.DeepEqual(log, want) {
		t.Fatalf("order = %v, want %v", log, want)
	}
}

// retryOnce asks for a second attempt and escalates the model on it.
type retryOnce struct{ core.Base }

func (retryOnce) Name() string { return "retry-once" }
func (retryOnce) Before(_ context.Context, s *core.Step) error {
	if s.State["escalate"] == true {
		s.Request.Model = "big"
	}
	return nil
}
func (retryOnce) After(_ context.Context, s *core.Step) error {
	if s.Attempt == 1 {
		s.State["escalate"] = true
		s.Retry = true
		s.Note("retry-once", "first answer rejected")
	}
	return nil
}

func TestRetryResetsRequestAndTraces(t *testing.T) {
	p := mock.New(mock.Text("bad"), mock.Text("good"))
	tr := &core.MemoryTracer{}
	e := &core.Engine{Provider: p, Layers: []core.Layer{retryOnce{}}, Tracer: tr}

	resp, err := e.Complete(context.Background(), core.Request{Model: "small"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "good" {
		t.Fatalf("content = %q, want good", resp.Message.Content)
	}
	reqs := p.Requests()
	if len(reqs) != 2 || reqs[0].Model != "small" || reqs[1].Model != "big" {
		t.Fatalf("models sent = %+v", reqs)
	}
	evs := tr.Events()
	if len(evs) != 2 || !evs[0].Retried || evs[1].Retried {
		t.Fatalf("events = %+v", evs)
	}
	if len(evs[0].Notes) != 1 || evs[1].Notes != nil {
		t.Fatalf("notes should be per attempt: %+v", evs)
	}
	if evs[0].RunID != evs[1].RunID {
		t.Fatal("attempts of one step must share a RunID")
	}
}

type alwaysRetry struct{ core.Base }

func (alwaysRetry) Name() string                                { return "always" }
func (alwaysRetry) After(_ context.Context, s *core.Step) error { s.Retry = true; return nil }

func TestMaxAttemptsIsAHardCap(t *testing.T) {
	p := mock.New(mock.Text("1"), mock.Text("2"), mock.Text("3"), mock.Text("4"))
	e := &core.Engine{Provider: p, Layers: []core.Layer{alwaysRetry{}}, MaxAttempts: 3}
	resp, err := e.Complete(context.Background(), core.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if got := len(p.Requests()); got != 3 {
		t.Fatalf("provider calls = %d, want 3", got)
	}
	if resp.Message.Content != "3" {
		t.Fatalf("should return the last attempt, got %q", resp.Message.Content)
	}
}

func TestProviderErrorWithoutRetryIsReturned(t *testing.T) {
	boom := errors.New("boom")
	e := &core.Engine{Provider: mock.New(mock.Reply{Err: boom})}
	if _, err := e.Complete(context.Background(), core.Request{}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
}

func TestCallerRequestIsNotMutated(t *testing.T) {
	mut := mutator{}
	req := core.Request{Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}}
	e := &core.Engine{Provider: mock.New(mock.Text("a")), Layers: []core.Layer{mut}}
	if _, err := e.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content != "q" {
		t.Fatalf("caller request changed: %+v", req)
	}
}

type mutator struct{ core.Base }

func (mutator) Name() string { return "mutator" }
func (mutator) Before(_ context.Context, s *core.Step) error {
	s.Request.Messages[0].Content = "changed"
	s.Request.Messages = append(s.Request.Messages, core.Message{Role: core.RoleSystem})
	return nil
}

func TestCancelledContextStopsBeforeProvider(t *testing.T) {
	p := mock.New(mock.Text("x"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	e := &core.Engine{Provider: p}
	if _, err := e.Complete(ctx, core.Request{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(p.Requests()) != 0 {
		t.Fatal("provider must not be called after cancellation")
	}
}

func TestReturnedUsageCoversEveryAttempt(t *testing.T) {
	first, second := mock.Text("bad"), mock.Text("good")
	first.Response.Usage = core.Usage{InputTokens: 100, OutputTokens: 10, CachedTokens: 40}
	second.Response.Usage = core.Usage{InputTokens: 120, OutputTokens: 20}
	e := &core.Engine{Provider: mock.New(first, second), Layers: []core.Layer{retryOnce{}}}

	resp, err := e.Complete(context.Background(), core.Request{Model: "small"})
	if err != nil {
		t.Fatal(err)
	}
	if want := (core.Usage{InputTokens: 220, OutputTokens: 30, CachedTokens: 40}); resp.Usage != want {
		t.Fatalf("usage = %+v, want %+v: a retry's tokens are paid for and must be reported", resp.Usage, want)
	}
	if second.Response.Usage.InputTokens != 120 {
		t.Fatal("the provider's own response object must not be edited")
	}
}

func TestJSONLTracerWritesOneSafeLinePerAttempt(t *testing.T) {
	var buf bytes.Buffer
	e := &core.Engine{Provider: mock.New(mock.Text("SECRET-ANSWER"), mock.Text("SECRET-ANSWER-2")), Layers: []core.Layer{retryOnce{}}, Tracer: core.NewJSONLTracer(&buf)}
	req := core.Request{Model: "m", Messages: []core.Message{{Role: core.RoleUser, Content: "SECRET-PROMPT"}}}
	if _, err := e.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("lines = %d: %s", len(lines), buf.String())
	}
	var first map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	if first["attempt"] != float64(1) || first["retried"] != true || first["input_tokens"] != float64(10) || first["model"] != "m" {
		t.Fatalf("first = %v", first)
	}
	if strings.Contains(buf.String(), "SECRET") {
		t.Fatal("a trace file must never contain prompt or response text")
	}
}

func TestTraceFileRecordsAuxiliarySpend(t *testing.T) {
	var buf bytes.Buffer
	e := &core.Engine{Provider: mock.New(mock.Text("a")), Layers: []core.Layer{auxLayer{}}, Tracer: core.NewJSONLTracer(&buf)}
	if _, err := e.Complete(context.Background(), core.Request{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	var rec map[string]any
	if err := json.Unmarshal(buf.Bytes(), &rec); err != nil {
		t.Fatal(err)
	}
	if rec["aux_input_tokens"] != float64(7) || rec["aux_output_tokens"] != float64(3) {
		t.Fatalf("trace = %v", rec)
	}
	if rec["input_tokens"] != float64(10) {
		t.Fatal("the supervised model's own usage must stay separate")
	}
}

type auxLayer struct{ core.Base }

func (auxLayer) Name() string { return "aux" }
func (auxLayer) After(_ context.Context, s *core.Step) error {
	s.AuxUsage.InputTokens += 7
	s.AuxUsage.OutputTokens += 3
	return nil
}
