package effort

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/rakuvo/burrow/core"
)

func run(t *testing.T, cfg Config, extra map[string]json.RawMessage, state map[string]any) map[string]json.RawMessage {
	t.Helper()
	req := core.Request{Model: "m", Extra: extra}
	s := &core.Step{Original: req.Clone(), Request: req.Clone(), State: state}
	if s.State == nil {
		s.State = map[string]any{}
	}
	if err := New(cfg).Before(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	return s.Request.Extra
}

func val(e map[string]json.RawMessage) string {
	var v string
	json.Unmarshal(e["reasoning_effort"], &v)
	return v
}

func TestSetsEffort(t *testing.T) {
	if got := val(run(t, Config{Effort: "minimal"}, nil, nil)); got != "minimal" {
		t.Fatalf("got %q", got)
	}
}

func TestZeroConfigChangesNothing(t *testing.T) {
	if e := run(t, Config{}, nil, nil); len(e) != 0 {
		t.Fatalf("got %v", e)
	}
}

func TestCallersChoiceWinsUnlessForced(t *testing.T) {
	caller := map[string]json.RawMessage{"reasoning_effort": json.RawMessage(`"high"`)}
	if got := val(run(t, Config{Effort: "minimal"}, caller, nil)); got != "high" {
		t.Fatalf("caller's effort overridden: %q", got)
	}
	if got := val(run(t, Config{Effort: "minimal", Force: true}, caller, nil)); got != "minimal" {
		t.Fatalf("force ignored: %q", got)
	}
}

func TestEscalatedCallGetsMoreThought(t *testing.T) {
	cfg := Config{Effort: "minimal", EscalatedEffort: "medium"}
	if got := val(run(t, cfg, nil, map[string]any{StateEscalated: true})); got != "medium" {
		t.Fatalf("got %q", got)
	}
	if got := val(run(t, cfg, nil, nil)); got != "minimal" {
		t.Fatalf("got %q", got)
	}
}

func TestOtherExtraFieldsSurviveAndOriginalIsUntouched(t *testing.T) {
	extra := map[string]json.RawMessage{"top_p": json.RawMessage(`0.5`)}
	req := core.Request{Extra: extra}
	s := &core.Step{Original: req.Clone(), Request: req.Clone(), State: map[string]any{}}
	New(Config{Effort: "low"}).Before(context.Background(), s)
	if string(s.Request.Extra["top_p"]) != "0.5" || len(s.Original.Extra) != 1 {
		t.Fatalf("request %v original %v", s.Request.Extra, s.Original.Extra)
	}
}

func TestValid(t *testing.T) {
	for _, v := range []string{"", "none", "minimal", "low", "medium", "high"} {
		if !Valid(v) {
			t.Errorf("%q rejected", v)
		}
	}
	if Valid("max") {
		t.Error("max accepted")
	}
}
