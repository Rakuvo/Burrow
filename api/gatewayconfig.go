package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/rakuvo/burrow/layers/effort"
)

// GatewayConfig is one organisation's layer settings. The control plane stores it, the
// gateway pulls it and builds its layer stack from it, so changing a setting in the
// dashboard changes the running gateway within one poll interval, with no restart.
// Every field's zero value means "off", except Guard, which is on unless set to false.
type GatewayConfig struct {
	Guard            *bool `json:"guard,omitempty"`              // validate and repair tool calls (default on)
	GuardPassThrough bool  `json:"guard_pass_through,omitempty"` // return a still-invalid call instead of failing
	ToolPolicy       bool  `json:"tool_policy,omitempty"`        // Prompt layer: tool-use instruction
	ToolPolicyAuto   bool  `json:"tool_policy_auto,omitempty"`   // Prompt layer: instruction only for models where it was measured to help

	Verify      VerifyConfig      `json:"verify,omitempty"`
	Context     ContextConfig     `json:"context,omitempty"`
	Loop        LoopConfig        `json:"loop,omitempty"`
	ModelPolicy ModelPolicyConfig `json:"model_policy,omitempty"`
	Memory      MemoryConfig      `json:"memory,omitempty"`
	Reasoning   ReasoningConfig   `json:"reasoning,omitempty"`
}

// ReasoningConfig sets how much hidden thinking reasoning models do. Lower effort cuts
// output tokens, the largest part of their bill. Accepted values depend on the model:
// "minimal" for gpt-5-nano and gpt-5-mini, "none" for newer ones.
type ReasoningConfig struct {
	Effort          string `json:"effort,omitempty"`           // none, minimal, low, medium, high
	EscalatedEffort string `json:"escalated_effort,omitempty"` // used once Model Policy escalates a call
	Force           bool   `json:"force,omitempty"`            // override an effort the caller set
}

type VerifyConfig struct {
	Model string `json:"model,omitempty"` // judge model; empty = Verify off
	Mode  string `json:"mode,omitempty"`  // veto (default), gate or both
	Scope string `json:"scope,omitempty"` // single (default): judge only one-request calls; all: every user turn
}

type ContextConfig struct {
	KeepObservations int    `json:"keep_observations,omitempty"`
	MaskStep         int    `json:"mask_step,omitempty"`
	MaxTokens        int    `json:"max_tokens,omitempty"`
	SummarizeModel   string `json:"summarize_model,omitempty"`
	FactsModel       string `json:"facts_model,omitempty"`
	KeepRelevant     int    `json:"keep_relevant,omitempty"`
	PruneTools       int    `json:"prune_tools,omitempty"`
	MaskWithinTurn   bool   `json:"mask_within_turn,omitempty"` // let masking reach results inside the latest user turn (long single-request agent runs)
}

type LoopConfig struct {
	Enabled      bool     `json:"enabled,omitempty"`
	MaxToolCalls int      `json:"max_tool_calls,omitempty"`
	AllowRepeat  []string `json:"allow_repeat,omitempty"`
	WarnOnly     bool     `json:"warn_only,omitempty"`
}

type ModelPolicyConfig struct {
	Cheap            string `json:"cheap,omitempty"`
	Strong           string `json:"strong,omitempty"`
	EscalateOnNoCall bool   `json:"escalate_on_no_call,omitempty"`
}

type MemoryConfig struct {
	Enabled bool   `json:"enabled,omitempty"` // needs the gateway to have a memory directory
	Model   string `json:"model,omitempty"`   // extractor model; empty = read-only memory
	// EmbedModel turns on meaning-based search and keeps each original message next to the facts
	// (the setup that scored best in the benchmark). SelectModel adds an evidence-selection call
	// per question; it needs EmbedModel.
	EmbedModel  string `json:"embed_model,omitempty"`
	SelectModel string `json:"select_model,omitempty"`
	TopK        int    `json:"top_k,omitempty"`
	Scope       string `json:"scope,omitempty"`
}

// GuardOn reports the Guard setting with its default.
func (c GatewayConfig) GuardOn() bool { return c.Guard == nil || *c.Guard }

// ParseGatewayConfig decodes and validates a config document. Unknown fields are an
// error, so a typo in a setting is caught when it is saved and not silently ignored.
func ParseGatewayConfig(raw []byte) (GatewayConfig, error) {
	var c GatewayConfig
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return GatewayConfig{}, fmt.Errorf("invalid config: %w", err)
	}
	return c, c.Validate()
}

func validModel(name string) bool {
	return len(name) <= 200 && !strings.ContainsAny(name, " \t\r\n")
}

// Validate checks settings that cannot work together.
func (c GatewayConfig) Validate() error {
	var errs []error
	bad := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }

	for _, v := range []struct {
		field, val string
	}{
		{"verify.model", c.Verify.Model}, {"context.summarize_model", c.Context.SummarizeModel},
		{"context.facts_model", c.Context.FactsModel}, {"model_policy.cheap", c.ModelPolicy.Cheap},
		{"model_policy.strong", c.ModelPolicy.Strong}, {"memory.model", c.Memory.Model},
		{"memory.embed_model", c.Memory.EmbedModel}, {"memory.select_model", c.Memory.SelectModel},
	} {
		if !validModel(v.val) {
			bad("%s is not a valid model name", v.field)
		}
	}
	for _, v := range []struct {
		field string
		val   int
	}{
		{"context.keep_observations", c.Context.KeepObservations}, {"context.mask_step", c.Context.MaskStep},
		{"context.max_tokens", c.Context.MaxTokens}, {"context.keep_relevant", c.Context.KeepRelevant},
		{"context.prune_tools", c.Context.PruneTools}, {"loop.max_tool_calls", c.Loop.MaxToolCalls}, {"memory.top_k", c.Memory.TopK},
	} {
		if v.val < 0 || v.val > 1_000_000 {
			bad("%s must be between 0 and 1000000", v.field)
		}
	}
	if c.ToolPolicy && c.ToolPolicyAuto {
		bad("tool_policy and tool_policy_auto are alternatives: set one")
	}
	if c.Verify.Model != "" {
		switch c.Verify.Mode {
		case "", "veto", "gate", "both":
		default:
			bad("verify.mode must be veto, gate or both, not %q", c.Verify.Mode)
		}
		switch c.Verify.Scope {
		case "", "single", "all":
		default:
			bad("verify.scope must be single or all, not %q", c.Verify.Scope)
		}
	} else if c.Verify.Mode != "" || c.Verify.Scope != "" {
		bad("verify.mode and verify.scope: needs verify.model")
	}
	if c.Context.FactsModel != "" && c.Context.KeepObservations <= 0 {
		bad("context.facts_model needs context.keep_observations")
	}
	if c.Context.KeepRelevant > 0 && c.Context.KeepObservations <= 0 {
		bad("context.keep_relevant needs context.keep_observations")
	}
	if c.Context.SummarizeModel != "" && c.Context.MaxTokens <= 0 {
		bad("context.summarize_model needs context.max_tokens")
	}
	if (c.ModelPolicy.Cheap == "") != (c.ModelPolicy.Strong == "") {
		bad("model_policy.cheap and model_policy.strong go together")
	}
	if c.ModelPolicy.Cheap != "" && c.ModelPolicy.Cheap == c.ModelPolicy.Strong {
		bad("model_policy.cheap and model_policy.strong must differ")
	}
	for _, v := range []struct{ field, val string }{
		{"reasoning.effort", c.Reasoning.Effort}, {"reasoning.escalated_effort", c.Reasoning.EscalatedEffort},
	} {
		if !effort.Valid(v.val) {
			bad("%s must be none, minimal, low, medium or high, not %q", v.field, v.val)
		}
	}
	if c.Reasoning.EscalatedEffort != "" && c.Reasoning.Effort == "" {
		bad("reasoning.escalated_effort needs reasoning.effort")
	}
	if c.Memory.SelectModel != "" && c.Memory.EmbedModel == "" {
		bad("memory.select_model needs memory.embed_model")
	}
	if (c.Memory.EmbedModel != "" || c.Memory.SelectModel != "") && !c.Memory.Enabled {
		bad("memory.embed_model and memory.select_model need memory.enabled")
	}
	if c.Memory.Model != "" && !c.Memory.Enabled {
		bad("memory.model needs memory.enabled")
	}
	return errors.Join(errs...)
}
