// Package gateway builds the layer stack of a Rakuvo gateway from its configuration. The terminal
// gateway and the hosted multi-tenant gateway share it, so a setting means the same thing in both.
package gateway

import (
	"context"
	"fmt"

	"github.com/rakuvo/burrow/api"
	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/layers/contextmgr"
	"github.com/rakuvo/burrow/layers/effort"
	"github.com/rakuvo/burrow/layers/guard"
	"github.com/rakuvo/burrow/layers/loopguard"
	"github.com/rakuvo/burrow/layers/memory"
	"github.com/rakuvo/burrow/layers/modelpolicy"
	"github.com/rakuvo/burrow/layers/prompt"
	"github.com/rakuvo/burrow/layers/toolprune"
	"github.com/rakuvo/burrow/layers/verify"
)

// BuildLayers turns a configuration into the layer stack, in the order that matters:
// Model Policy first (After hooks run in reverse, so it then sees the retry decisions
// of every layer after it), then Memory, Context, pruning, Loop, Prompt, Guard, Verify.
// memStore is the gateway's shared memory store, or nil when it has no memory directory;
// reusing it across rebuilds keeps remembered facts through a config change.
func BuildLayers(cfg api.GatewayConfig, p core.Provider, memStore *memory.Store, judgePrompt string) ([]core.Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var ls []core.Layer
	if cfg.ModelPolicy.Cheap != "" {
		ls = append(ls, modelpolicy.New(modelpolicy.Config{
			Cheap: cfg.ModelPolicy.Cheap, Strong: cfg.ModelPolicy.Strong, EscalateOnNoCall: cfg.ModelPolicy.EscalateOnNoCall}))
	}
	if r := cfg.Reasoning; r.Effort != "" {
		ls = append(ls, effort.New(effort.Config{Effort: r.Effort, EscalatedEffort: r.EscalatedEffort, Force: r.Force}))
	}
	if cfg.Memory.Enabled {
		if memStore == nil {
			return nil, fmt.Errorf("memory is enabled but this gateway has no -memory-dir")
		}
		topK := cfg.Memory.TopK
		if topK == 0 {
			topK = 6
		}
		scope := cfg.Memory.Scope
		if scope == "" {
			scope = "default"
		}
		mc := memory.Config{Store: memStore, TopK: topK, DefaultScope: scope, Async: true}
		if cfg.Memory.Model != "" {
			mc.Extractor = memory.NewLLMExtractor(p, cfg.Memory.Model, "", 0, 0)
		}
		if cfg.Memory.EmbedModel != "" {
			ep, ok := p.(embedProvider)
			if !ok {
				return nil, fmt.Errorf("memory.embed_model needs an upstream with an embeddings endpoint")
			}
			mc.Embedder = providerEmbedder{ep, cfg.Memory.EmbedModel}
			if cfg.Memory.SelectModel != "" {
				mc.Selector = memory.NewLLMSelector(p, cfg.Memory.SelectModel, 0)
			}
		}
		ls = append(ls, memory.New(mc))
	}
	c := cfg.Context
	if c.KeepObservations > 0 || c.MaxTokens > 0 {
		cc := contextmgr.Config{KeepLastObservations: c.KeepObservations, MaskStep: c.MaskStep,
			MaxInputTokens: c.MaxTokens, KeepRecentTurns: 1, KeepRelevant: c.KeepRelevant, MaskWithinTurn: c.MaskWithinTurn}
		if c.FactsModel != "" {
			cc.Facts = contextmgr.NewLLMFactExtractor(p, c.FactsModel, "", 0, 0)
		}
		if c.SummarizeModel != "" {
			cc.Summarizer = contextmgr.NewLLMSummarizer(p, c.SummarizeModel, "", 0, 0)
		}
		ls = append(ls, contextmgr.New(cc))
	}
	if c.PruneTools > 0 {
		ls = append(ls, toolprune.New(toolprune.Config{Keep: c.PruneTools}))
	}
	if l := cfg.Loop; l.Enabled || l.MaxToolCalls > 0 {
		lc := loopguard.Config{}
		if l.Enabled {
			lc = loopguard.Defaults()
		}
		lc.MaxToolCalls = l.MaxToolCalls
		if l.WarnOnly {
			lc.Action = loopguard.WarnOnly
		}
		lc.AllowRepeat = append(lc.AllowRepeat, l.AllowRepeat...)
		ls = append(ls, loopguard.New(lc))
	}
	if cfg.ToolPolicy || cfg.ToolPolicyAuto {
		ls = append(ls, prompt.New(prompt.Config{Auto: cfg.ToolPolicyAuto && !cfg.ToolPolicy}))
	}
	if cfg.GuardOn() {
		gc := guard.Config{}
		if cfg.GuardPassThrough {
			gc.OnFailure = guard.PassThrough
		}
		ls = append(ls, guard.New(gc))
	}
	if cfg.Verify.Model != "" {
		var vc verify.Config
		switch cfg.Verify.Mode {
		case "gate":
			vc.Gate = true
		case "both":
			vc.Gate, vc.Veto = true, true
		default: // veto is the validated mode
			vc.Veto = true
		}
		vc.SingleRequestOnly = cfg.Verify.Scope != "all"
		ls = append(ls, verify.New(verify.NewJudge(p, cfg.Verify.Model, judgePrompt, 0), vc))
	}
	return ls, nil
}

// embedProvider is a provider that can also create embeddings (the OpenAI-compatible one can).
type embedProvider interface {
	Embed(ctx context.Context, model string, texts []string) ([][]float32, core.Usage, error)
}

type providerEmbedder struct {
	p     embedProvider
	model string
}

func (e providerEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	v, _, err := e.p.Embed(ctx, e.model, texts)
	return v, err
}
