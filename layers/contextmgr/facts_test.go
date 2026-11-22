package contextmgr

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/rakuvo/burrow/core"
)

type fakeFacts struct {
	mu    sync.Mutex
	calls int
	out   string
	err   error
}

func (f *fakeFacts) Extract(_ context.Context, tool, args, output string) (string, core.Usage, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()
	return f.out, core.Usage{InputTokens: 50, OutputTokens: 10}, f.err
}

func TestFactsAppearInPlaceholder(t *testing.T) {
	f := &fakeFacts{out: "order id 4412, total 87.50"}
	cfg := Config{KeepLastObservations: 2, KeepRecentTurns: 1, Facts: f}
	got, _ := run(t, cfg, core.Request{Messages: conversation(6, 1000)})
	var n int
	for _, r := range toolResults(got.Messages) {
		if masked(r) {
			n++
			if !strings.Contains(r, "Key facts: order id 4412, total 87.50") {
				t.Fatalf("placeholder lacks facts: %q", r)
			}
		}
	}
	if n == 0 {
		t.Fatal("nothing was masked")
	}
}

func TestFactsExtractedOncePerResult(t *testing.T) {
	f := &fakeFacts{out: "x"}
	l := New(Config{KeepLastObservations: 2, KeepRecentTurns: 1, Facts: f})
	req := core.Request{Messages: conversation(6, 1000)}
	for i := 0; i < 3; i++ {
		s := &core.Step{Original: req.Clone(), Request: req.Clone()}
		if err := l.Before(context.Background(), s); err != nil {
			t.Fatal(err)
		}
	}
	first := f.calls
	if first == 0 || first > 4 {
		t.Fatalf("expected one call per masked result (<=4), got %d", first)
	}
	s := &core.Step{Original: req.Clone(), Request: req.Clone()}
	l.Before(context.Background(), s)
	if f.calls != first {
		t.Fatalf("repeat request called the extractor again: %d -> %d", first, f.calls)
	}
}

func TestFactsCostReportedAndFailureFallsBack(t *testing.T) {
	f := &fakeFacts{out: "k"}
	_, s := run(t, Config{KeepLastObservations: 2, KeepRecentTurns: 1, Facts: f}, core.Request{Messages: conversation(6, 1000)})
	if s.AuxUsage.InputTokens == 0 {
		t.Fatal("extractor spend not reported")
	}
	bad := &fakeFacts{err: errors.New("down")}
	got, _ := run(t, Config{KeepLastObservations: 2, KeepRecentTurns: 1, Facts: bad}, core.Request{Messages: conversation(6, 1000)})
	for _, r := range toolResults(got.Messages) {
		if masked(r) && strings.Contains(r, "Key facts") {
			t.Fatal("failed extraction must not add a facts note")
		}
	}
}

func TestNothingNeededGivesPlainPlaceholder(t *testing.T) {
	f := &fakeFacts{out: "Nothing needed."}
	got, _ := run(t, Config{KeepLastObservations: 2, KeepRecentTurns: 1, Facts: f}, core.Request{Messages: conversation(6, 1000)})
	for _, r := range toolResults(got.Messages) {
		if strings.Contains(r, "Key facts") {
			t.Fatalf("got %q", r)
		}
	}
}

func relevanceConv() []core.Message {
	msgs := []core.Message{sys("s")}
	topics := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}
	for i, tp := range topics {
		id := string(rune('a' + i))
		msgs = append(msgs, user("look up "+tp), core.Message{Role: core.RoleAssistant,
			ToolCalls: []core.ToolCall{{ID: id, Name: "lookup", Arguments: `{"q":"` + tp + `"}`}}},
			result(id, tp+" "+strings.Repeat("data ", 100)))
	}
	msgs = append(msgs, user("go back to the beta record and change it"))
	return msgs
}

func TestRelevantOldResultSurvivesMasking(t *testing.T) {
	cfg := Config{KeepLastObservations: 2, KeepRelevant: 1, KeepRecentTurns: 1}
	got, _ := run(t, cfg, core.Request{Messages: relevanceConv()})
	res := toolResults(got.Messages)
	if masked(res[1]) { // beta
		t.Fatalf("the result the user asks about was masked: %q", res[1][:40])
	}
	if !masked(res[0]) || !masked(res[2]) {
		t.Fatal("irrelevant old results should still be masked")
	}
}

func TestRelevanceOffMasksByAgeOnly(t *testing.T) {
	got, _ := run(t, Config{KeepLastObservations: 2, KeepRecentTurns: 1}, core.Request{Messages: relevanceConv()})
	if !masked(toolResults(got.Messages)[1]) {
		t.Fatal("without KeepRelevant the old result must be masked")
	}
}
