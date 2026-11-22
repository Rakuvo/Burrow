package modelpolicy

import (
	"context"
	"errors"
	"testing"

	"github.com/rakuvo/burrow/core"
)

// scripted returns the replies in order and records the model of each request.
type scripted struct {
	replies []func() (*core.Response, error)
	models  []string
}

func (*scripted) Name() string { return "scripted" }
func (p *scripted) Complete(_ context.Context, r core.Request) (*core.Response, error) {
	p.models = append(p.models, r.Model)
	f := p.replies[len(p.replies)-1]
	if len(p.replies) > 1 {
		f, p.replies = p.replies[0], p.replies[1:]
	}
	return f()
}

func text(s string) func() (*core.Response, error) {
	return func() (*core.Response, error) {
		return &core.Response{Message: core.Message{Role: core.RoleAssistant, Content: s}, FinishReason: "stop"}, nil
	}
}

func call() func() (*core.Response, error) {
	return func() (*core.Response, error) {
		return &core.Response{Message: core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "1", Name: "f", Arguments: "{}"}}}}, nil
	}
}

func fail(err error) func() (*core.Response, error) {
	return func() (*core.Response, error) { return nil, err }
}

// rejecter is a stand-in for Guard or Verify: it asks for a retry on the first attempt only.
type rejecter struct{ times int }

func (*rejecter) Name() string                             { return "rejecter" }
func (*rejecter) Before(context.Context, *core.Step) error { return nil }
func (r *rejecter) After(_ context.Context, s *core.Step) error {
	if s.Attempt <= r.times && s.Response != nil {
		s.Retry = true
	}
	return nil
}

func run(t *testing.T, p *scripted, cfg Config, model string, extra ...core.Layer) (*core.Response, []string) {
	t.Helper()
	tr := &core.MemoryTracer{}
	e := &core.Engine{Provider: p, Layers: append([]core.Layer{New(cfg)}, extra...), Tracer: tr}
	resp, err := e.Complete(context.Background(), core.Request{Model: model, Messages: []core.Message{{Role: core.RoleUser, Content: "hi"}},
		Tools: []core.ToolSpec{{Name: "f"}}})
	_ = err
	var notes []string
	for _, ev := range tr.Events() {
		for _, n := range ev.Notes {
			notes = append(notes, n.Message)
		}
	}
	return resp, notes
}

var cfg = Config{Cheap: "small", Strong: "big"}

func TestCheapModelServesWhenNothingFails(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){call()}}
	run(t, p, cfg, "big")
	if len(p.models) != 1 || p.models[0] != "small" {
		t.Fatalf("models %v", p.models)
	}
}

func TestEscalatesWhenAnotherLayerRejects(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){call()}}
	_, notes := run(t, p, cfg, "big", &rejecter{times: 1})
	if len(p.models) != 2 || p.models[0] != "small" || p.models[1] != "big" {
		t.Fatalf("models %v", p.models)
	}
	if len(notes) == 0 {
		t.Fatal("escalation not recorded in the trace")
	}
}

func TestStaysOnStrongModelAfterEscalation(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){call()}}
	run(t, p, cfg, "big", &rejecter{times: 2})
	if len(p.models) != 3 || p.models[1] != "big" || p.models[2] != "big" {
		t.Fatalf("models %v", p.models)
	}
}

func TestProviderErrorFallsBackToStrong(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){fail(errors.New("503")), call()}}
	resp, _ := run(t, p, cfg, "big")
	if resp == nil || len(p.models) != 2 || p.models[1] != "big" {
		t.Fatalf("models %v resp %v", p.models, resp)
	}
}

func TestProviderErrorNotEscalatedWhenDisabled(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){fail(errors.New("503"))}}
	c := cfg
	c.DisableOnError = true
	resp, _ := run(t, p, c, "big")
	if resp != nil || len(p.models) != 1 {
		t.Fatalf("models %v", p.models)
	}
}

func TestEmptyResponseEscalates(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){text(""), call()}}
	run(t, p, cfg, "big")
	if len(p.models) != 2 || p.models[1] != "big" {
		t.Fatalf("models %v", p.models)
	}
}

func TestTextAnswerAcceptedByDefaultEscalatedWhenAsked(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){text("no tool applies")}}
	run(t, p, cfg, "big")
	if len(p.models) != 1 {
		t.Fatalf("a text answer escalated by default: %v", p.models)
	}
	p = &scripted{replies: []func() (*core.Response, error){text("no tool applies"), call()}}
	c := cfg
	c.EscalateOnNoCall = true
	run(t, p, c, "big")
	if len(p.models) != 2 || p.models[1] != "big" {
		t.Fatalf("models %v", p.models)
	}
}

func TestOtherModelsPassThroughAndZeroConfigDoesNothing(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){call()}}
	run(t, p, cfg, "something-else", &rejecter{times: 1})
	if p.models[0] != "something-else" || p.models[1] != "something-else" {
		t.Fatalf("models %v", p.models)
	}
	p = &scripted{replies: []func() (*core.Response, error){call()}}
	run(t, p, Config{}, "big")
	if p.models[0] != "big" {
		t.Fatalf("models %v", p.models)
	}
}

func TestNoEscalationOnLastAttempt(t *testing.T) {
	p := &scripted{replies: []func() (*core.Response, error){call()}}
	tr := &core.MemoryTracer{}
	e := &core.Engine{Provider: p, MaxAttempts: 1, Layers: []core.Layer{New(cfg), &rejecter{times: 5}}, Tracer: tr}
	e.Complete(context.Background(), core.Request{Model: "big", Messages: []core.Message{{Role: core.RoleUser, Content: "x"}}})
	if len(p.models) != 1 {
		t.Fatalf("models %v", p.models)
	}
}
