// Package prompt is the Prompt layer. It adds a short, fixed instruction to
// requests that offer tools, aimed at the commonest way models misuse them: they
// answer from memory, do the arithmetic themselves, or ask a question instead of
// calling the tool that was provided for exactly that.
//
// The instruction is appended to the caller's own system message (or becomes one),
// so it adds no model call and no latency, only a few input tokens.
package prompt

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/rakuvo/burrow/core"
)

// DefaultToolPolicy is the instruction added when Config.ToolPolicy is empty.
//
// It deliberately keeps the escape hatch ("if none of the tools is relevant,
// answer normally"), because a policy that forces a tool call on every turn would
// trade one reliability problem for another.
const DefaultToolPolicy = `You have access to tools (functions). When a tool can answer the user's request, or perform part of it, call the tool instead of answering from memory or working it out yourself. Do not describe what you would do; make the call. If a request needs several independent calls, make all of them in the same response. Take argument values from the user's request, and infer an obvious value when the request implies one. Ask a clarifying question only when a required argument is missing and cannot reasonably be inferred. If none of the tools is relevant to the request, answer normally without calling any tool.`

// Config tunes the layer. The zero value uses DefaultToolPolicy.
type Config struct {
	// ToolPolicy replaces DefaultToolPolicy.
	ToolPolicy string
	// Auto applies the policy only to models where it was measured to help (see PresetFor).
	Auto bool
}

// Layer implements core.Layer. It keeps no per-request state.
type Layer struct {
	policy string
	auto   bool
}

func New(cfg Config) *Layer {
	p := strings.TrimSpace(cfg.ToolPolicy)
	if p == "" {
		p = DefaultToolPolicy
	}
	return &Layer{policy: p, auto: cfg.Auto}
}

// presets holds what the BFCL multi-turn experiments found per model family. A longer
// prefix wins over a shorter one.
var presets = []struct {
	prefix string
	on     bool
	why    string
}{
	{"gpt-5-nano", true, "measured +6.4 points on BFCL multi-turn"},
	{"gpt-5-mini", true, "measured +16.2 points on BFCL multi-turn"},
	{"gpt-4.1-mini", false, "measured no significant gain"},
	{"gpt-4.1", false, "measured no gain"},
	{"gpt-4o-mini", false, "measured no significant gain"},
}

// PresetFor says whether the policy should be applied to model, whether that is a
// measured result, and why. Models nobody has measured get the policy, flagged unmeasured.
func PresetFor(model string) (on, measured bool, why string) {
	best := -1
	for i, p := range presets {
		if strings.HasPrefix(model, p.prefix) && (best < 0 || len(p.prefix) > len(presets[best].prefix)) {
			best = i
		}
	}
	if best < 0 {
		return true, false, "not measured for this model"
	}
	return presets[best].on, true, presets[best].why
}

func (*Layer) Name() string { return "prompt" }

// Before adds the policy. The engine hands every attempt a fresh copy of the
// caller's request, so the policy is never added twice on a retry.
func (l *Layer) Before(_ context.Context, s *core.Step) error {
	s.State[core.StateToolPolicyApplied] = false
	if len(s.Request.Tools) == 0 || toolsDisabled(s.Request) {
		return nil
	}
	if off, _ := s.State[core.StateNoToolPolicy].(bool); off {
		return nil // another layer asked for this attempt to run without the policy
	}
	if l.auto {
		if on, _, why := PresetFor(s.Request.Model); !on {
			s.Note(l.Name(), "tool-use policy not added for "+s.Request.Model+": "+why)
			return nil
		}
	}
	msgs := s.Request.Messages
	if len(msgs) > 0 && msgs[0].Role == core.RoleSystem {
		if strings.Contains(msgs[0].Content, l.policy) {
			return nil // already applied upstream of us
		}
		sep := "\n\n"
		if strings.TrimSpace(msgs[0].Content) == "" {
			sep = ""
		}
		msgs[0].Content += sep + l.policy
	} else {
		s.Request.Messages = append([]core.Message{{Role: core.RoleSystem, Content: l.policy}}, msgs...)
	}
	s.State[core.StateToolPolicyApplied] = true
	s.Note(l.Name(), "added the tool-use policy")
	return nil
}

func (*Layer) After(context.Context, *core.Step) error { return nil }

// toolsDisabled reports tool_choice "none": the caller has forbidden calls, so an
// instruction to make them would contradict the request.
func toolsDisabled(req core.Request) bool {
	raw, ok := req.Extra["tool_choice"]
	if !ok {
		return false
	}
	var mode string
	return json.Unmarshal(raw, &mode) == nil && mode == "none"
}
