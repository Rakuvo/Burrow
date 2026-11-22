package prompt_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/layers/guard"
	"github.com/rakuvo/burrow/layers/prompt"
	"github.com/rakuvo/burrow/providers/mock"
)

var tool = core.ToolSpec{Name: "f", Schema: `{"type":"object"}`}

func run(t *testing.T, cfg prompt.Config, req core.Request) (core.Request, *core.MemoryTracer) {
	t.Helper()
	p, tr := mock.New(mock.Text("ok")), &core.MemoryTracer{}
	e := &core.Engine{Provider: p, Layers: []core.Layer{prompt.New(cfg)}, Tracer: tr}
	if _, err := e.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	return p.Requests()[0], tr
}

func TestPolicyBecomesTheSystemMessageWhenThereIsNone(t *testing.T) {
	got, tr := run(t, prompt.Config{}, core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}})
	if len(got.Messages) != 2 || got.Messages[0].Role != core.RoleSystem || got.Messages[0].Content != prompt.DefaultToolPolicy {
		t.Fatalf("messages = %+v", got.Messages)
	}
	if got.Messages[1].Content != "q" {
		t.Fatal("the user's message must follow unchanged")
	}
	if n := tr.Events()[0].Notes; len(n) != 1 || n[0].Layer != "prompt" {
		t.Fatalf("the trace should record that the policy was added: %+v", n)
	}
}

func TestPolicyIsAppendedToTheCallersSystemMessage(t *testing.T) {
	got, _ := run(t, prompt.Config{}, core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{
		{Role: core.RoleSystem, Content: "You are a banking assistant."}, {Role: core.RoleUser, Content: "q"}}})
	if len(got.Messages) != 2 {
		t.Fatalf("no extra message should be added: %+v", got.Messages)
	}
	if !strings.HasPrefix(got.Messages[0].Content, "You are a banking assistant.\n\n") || !strings.HasSuffix(got.Messages[0].Content, prompt.DefaultToolPolicy) {
		t.Fatalf("the caller's instructions must come first and survive intact: %q", got.Messages[0].Content)
	}
}

func TestNothingIsAddedWhenNoToolsAreOffered(t *testing.T) {
	got, tr := run(t, prompt.Config{}, core.Request{Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}})
	if len(got.Messages) != 1 || len(tr.Events()[0].Notes) != 0 {
		t.Fatalf("a tool policy on a tool-less request is noise: %+v", got.Messages)
	}
}

func TestNothingIsAddedWhenToolChoiceIsNone(t *testing.T) {
	got, _ := run(t, prompt.Config{}, core.Request{
		Tools:    []core.ToolSpec{tool},
		Extra:    map[string]json.RawMessage{"tool_choice": json.RawMessage(`"none"`)},
		Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}})
	if len(got.Messages) != 1 {
		t.Fatalf("telling the model to call tools when the caller forbade it would contradict the request: %+v", got.Messages)
	}
}

func TestPolicyIsNotAppliedTwice(t *testing.T) {
	got, _ := run(t, prompt.Config{}, core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{
		{Role: core.RoleSystem, Content: "x\n\n" + prompt.DefaultToolPolicy}, {Role: core.RoleUser, Content: "q"}}})
	if strings.Count(got.Messages[0].Content, prompt.DefaultToolPolicy) != 1 {
		t.Fatalf("policy duplicated: %q", got.Messages[0].Content)
	}
}

func TestCustomPolicy(t *testing.T) {
	got, _ := run(t, prompt.Config{ToolPolicy: "  Always use tools.  "}, core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}})
	if got.Messages[0].Content != "Always use tools." {
		t.Fatalf("content = %q", got.Messages[0].Content)
	}
}

func TestCallersRequestIsNeverEdited(t *testing.T) {
	req := core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{
		{Role: core.RoleSystem, Content: "sys"}, {Role: core.RoleUser, Content: "q"}}}
	run(t, prompt.Config{}, req)
	if req.Messages[0].Content != "sys" || len(req.Messages) != 2 {
		t.Fatalf("caller's request was changed: %+v", req.Messages)
	}
}

// With the guard retrying, each attempt must carry the policy exactly once, and
// the guard's corrective message must still come last.
func TestPolicyOnEveryAttemptOfARetry(t *testing.T) {
	bad := mock.Reply{Response: &core.Response{Message: core.Message{Role: core.RoleAssistant,
		ToolCalls: []core.ToolCall{{ID: "1", Name: "nope", Arguments: "{}"}}}}}
	good := mock.Reply{Response: &core.Response{Message: core.Message{Role: core.RoleAssistant,
		ToolCalls: []core.ToolCall{{ID: "2", Name: "f", Arguments: "{}"}}}}}
	p := mock.New(bad, good)
	e := &core.Engine{Provider: p, Layers: []core.Layer{prompt.New(prompt.Config{}), guard.New(guard.Config{})}}
	if _, err := e.Complete(context.Background(), core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}}); err != nil {
		t.Fatal(err)
	}
	reqs := p.Requests()
	if len(reqs) != 2 {
		t.Fatalf("attempts = %d", len(reqs))
	}
	for i, r := range reqs {
		if r.Messages[0].Role != core.RoleSystem || strings.Count(r.Messages[0].Content, prompt.DefaultToolPolicy) != 1 {
			t.Errorf("attempt %d: policy missing or duplicated: %+v", i+1, r.Messages[0])
		}
	}
	if last := reqs[1].Messages[len(reqs[1].Messages)-1]; last.Role != core.RoleUser || !strings.Contains(last.Content, "invalid tool calls") {
		t.Fatalf("guard feedback should be last on the retry: %+v", last)
	}
}

func TestPolicyIsSkippedWhenAnotherLayerAsksForAnAttemptWithoutIt(t *testing.T) {
	l := prompt.New(prompt.Config{})
	s := &core.Step{State: map[string]any{core.StateNoToolPolicy: true}, Request: core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}}}
	if err := l.Before(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	if len(s.Request.Messages) != 1 {
		t.Fatalf("the policy must be skipped on request of another layer: %+v", s.Request.Messages)
	}
}

func TestPolicyReportsWhetherItWasApplied(t *testing.T) {
	l := prompt.New(prompt.Config{})
	s := &core.Step{State: map[string]any{}, Request: core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}}}
	_ = l.Before(context.Background(), s)
	if applied, _ := s.State[core.StateToolPolicyApplied].(bool); !applied {
		t.Fatal("policy applied but not reported")
	}
	s.State[core.StateNoToolPolicy] = true
	s.Request = core.Request{Tools: []core.ToolSpec{tool}, Messages: []core.Message{{Role: core.RoleUser, Content: "q"}}}
	_ = l.Before(context.Background(), s)
	if applied, _ := s.State[core.StateToolPolicyApplied].(bool); applied {
		t.Fatal("a skipped attempt must not report the policy as applied")
	}
}

func TestAutoAppliesOnlyWhereItWasMeasuredToHelp(t *testing.T) {
	for model, want := range map[string]bool{
		"gpt-5-nano-2025-08-07": true, "gpt-5-mini": true,
		"gpt-4.1-mini-2025-04-14": false, "gpt-4.1-2025-04-14": false, "gpt-4o-mini": false,
		"some-new-model": true, // unmeasured models get the policy
	} {
		s := &core.Step{State: map[string]any{}, Request: core.Request{Model: model,
			Tools:    []core.ToolSpec{{Name: "t", Schema: `{}`}},
			Messages: []core.Message{{Role: core.RoleUser, Content: "hi"}}}}
		if err := prompt.New(prompt.Config{Auto: true}).Before(context.Background(), s); err != nil {
			t.Fatal(err)
		}
		got, _ := s.State[core.StateToolPolicyApplied].(bool)
		if got != want {
			t.Errorf("%s: applied=%v, want %v", model, got, want)
		}
	}
}

func TestPresetForPrefersTheLongestMatchAndFlagsUnmeasured(t *testing.T) {
	if on, measured, _ := prompt.PresetFor("gpt-4.1-mini-x"); on || !measured {
		t.Errorf("gpt-4.1-mini: on=%v measured=%v", on, measured)
	}
	if on, measured, _ := prompt.PresetFor("claude-x"); !on || measured {
		t.Errorf("unknown model: on=%v measured=%v", on, measured)
	}
}
