// Package presets names the model providers Rakuvo knows how to reach, so a
// customer writes `-provider deepseek` instead of looking up a base URL.
package presets

import (
	"fmt"
	"sort"
	"strings"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/providers/anthropic"
	"github.com/rakuvo/burrow/providers/openai"
)

// Spec describes one provider.
type Spec struct {
	Native         bool   // true: not OpenAI-compatible, uses its own adapter (Claude)
	BaseURL        string // API root
	KeyEnv         string // env var that usually holds the key ("" = none needed)
	MaxTokensField string // OpenAI-format output-limit field this provider accepts
	OmitTemp       bool
}

const (
	legacy     = openai.MaxTokensLegacy
	completion = openai.MaxTokensCompletion
)

var table = map[string]Spec{
	"openai":     {BaseURL: "https://api.openai.com/v1", KeyEnv: "OPENAI_API_KEY", MaxTokensField: completion},
	"anthropic":  {Native: true, BaseURL: anthropic.DefaultBaseURL, KeyEnv: "ANTHROPIC_API_KEY"},
	"deepseek":   {BaseURL: "https://api.deepseek.com/v1", KeyEnv: "DEEPSEEK_API_KEY", MaxTokensField: legacy},
	"openrouter": {BaseURL: "https://openrouter.ai/api/v1", KeyEnv: "OPENROUTER_API_KEY", MaxTokensField: legacy},
	"gemini":     {BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai", KeyEnv: "GEMINI_API_KEY", MaxTokensField: legacy},
	"groq":       {BaseURL: "https://api.groq.com/openai/v1", KeyEnv: "GROQ_API_KEY", MaxTokensField: legacy},
	"together":   {BaseURL: "https://api.together.xyz/v1", KeyEnv: "TOGETHER_API_KEY", MaxTokensField: legacy},
	"fireworks":  {BaseURL: "https://api.fireworks.ai/inference/v1", KeyEnv: "FIREWORKS_API_KEY", MaxTokensField: legacy},
	"mistral":    {BaseURL: "https://api.mistral.ai/v1", KeyEnv: "MISTRAL_API_KEY", MaxTokensField: legacy},
	"xai":        {BaseURL: "https://api.x.ai/v1", KeyEnv: "XAI_API_KEY", MaxTokensField: legacy},
	// Open-source servers you run yourself: no key, local address.
	"ollama":   {BaseURL: "http://localhost:11434/v1", MaxTokensField: legacy},
	"vllm":     {BaseURL: "http://localhost:8000/v1", MaxTokensField: legacy},
	"lmstudio": {BaseURL: "http://localhost:1234/v1", MaxTokensField: legacy},
	// Any other OpenAI-compatible server: give -upstream.
	"custom": {MaxTokensField: legacy},
}

// Lookup returns the preset for a name (case-insensitive).
func Lookup(name string) (Spec, bool) {
	s, ok := table[strings.ToLower(strings.TrimSpace(name))]
	return s, ok
}

// Names lists the known providers.
func Names() []string {
	out := make([]string, 0, len(table))
	for n := range table {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Options override a preset. Empty fields keep the preset's value.
type Options struct {
	BaseURL        string
	APIKey         string
	MaxTokensField string
	OmitTemp       *bool
}

// New builds the provider for a named preset.
func New(name string, o Options) (core.Provider, error) {
	s, ok := Lookup(name)
	if !ok {
		return nil, fmt.Errorf("unknown provider %q (known: %s)", name, strings.Join(Names(), ", "))
	}
	base := s.BaseURL
	if o.BaseURL != "" {
		base = o.BaseURL
	}
	if base == "" {
		return nil, fmt.Errorf("provider %q needs a base URL (-upstream)", name)
	}
	if s.Native {
		return anthropic.New(anthropic.Config{Name: strings.ToLower(name), BaseURL: base, APIKey: o.APIKey}), nil
	}
	field := s.MaxTokensField
	if o.MaxTokensField != "" {
		field = o.MaxTokensField
	}
	omit := s.OmitTemp
	if o.OmitTemp != nil {
		omit = *o.OmitTemp
	}
	return openai.New(openai.Config{Name: strings.ToLower(name), BaseURL: base, APIKey: o.APIKey, MaxTokensField: field, OmitTemperature: omit}), nil
}
