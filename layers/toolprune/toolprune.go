// Package toolprune is the tool-schema pruning part of the Context layer. Agents often
// send every tool definition on every call, which can be most of the prompt and gives
// the model more wrong tools to pick from. This layer sends only the Keep tools that
// match the latest user message best, plus every tool the conversation has already
// called, so a tool in use never disappears.
//
// Matching is deterministic and local: tokens of the latest user message against the
// tokens of each tool's name and description, weighted by how rare each token is across
// the tool set (the IDF part of BM25). No model call, no added latency. The kept tools
// stay in their original order. A request with no more than MinTools tools, no user
// message, or no match at all is left as it is.
package toolprune

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/internal/textmatch"
)

// Config selects what the layer does. The zero value changes nothing.
type Config struct {
	// Keep is how many best-matching tools to send, before adding tools already used
	// in the conversation. Zero turns the layer off.
	Keep int
	// MinTools: requests with this many tools or fewer are left alone, since pruning a
	// short list saves little and risks the most. Zero means Keep*2.
	MinTools int
}

// Layer implements core.Layer.
type Layer struct{ cfg Config }

func New(cfg Config) *Layer {
	if cfg.Keep > 0 && cfg.MinTools <= 0 {
		cfg.MinTools = cfg.Keep * 2
	}
	return &Layer{cfg: cfg}
}

func (*Layer) Name() string { return "toolprune" }

func (*Layer) After(context.Context, *core.Step) error { return nil }

func (l *Layer) Before(_ context.Context, s *core.Step) error {
	if l.cfg.Keep <= 0 {
		return nil
	}
	req := &s.Request
	if len(req.Tools) <= l.cfg.MinTools {
		return nil
	}
	query := tokens(lastUser(req.Messages))
	if len(query) == 0 {
		return nil
	}
	must := map[string]bool{}
	for _, m := range req.Messages {
		for _, c := range m.ToolCalls {
			must[c.Name] = true
		}
	}
	if n := forcedTool(req.Extra); n != "" {
		must[n] = true
	}

	scores := score(req.Tools, query)
	matched := 0
	for _, sc := range scores {
		if sc > 0 {
			matched++
		}
	}
	if matched == 0 {
		return nil
	}
	order := make([]int, len(req.Tools))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
	keep := make([]bool, len(req.Tools))
	for i, t := range req.Tools {
		keep[i] = must[t.Name]
	}
	for _, i := range order[:l.cfg.Keep] {
		keep[i] = true
	}
	var kept []core.ToolSpec
	for i, t := range req.Tools {
		if keep[i] {
			kept = append(kept, t)
		}
	}
	if len(kept) == len(req.Tools) {
		return nil
	}
	before, after := toolChars(req.Tools), toolChars(kept)
	req.Tools = kept
	s.Note(l.Name(), fmt.Sprintf("sent %d of %d tools; about %d input tokens saved", len(kept), len(s.Original.Tools), (before-after)/4))
	return nil
}

// score returns one relevance score per tool: the summed IDF of the query tokens found
// in the tool, a name match counting double.
func score(tools []core.ToolSpec, query map[string]bool) []float64 {
	names := make([]map[string]bool, len(tools))
	descs := make([]map[string]bool, len(tools))
	df := map[string]int{}
	for i, t := range tools {
		names[i] = tokens(t.Name)
		descs[i] = tokens(t.Description)
		seen := map[string]bool{}
		for w := range names[i] {
			seen[w] = true
		}
		for w := range descs[i] {
			seen[w] = true
		}
		for w := range seen {
			df[w]++
		}
	}
	n := float64(len(tools))
	out := make([]float64, len(tools))
	for i := range tools {
		for w := range query {
			d := df[w]
			if d == 0 {
				continue
			}
			idf := math.Log(1 + (n-float64(d)+0.5)/(float64(d)+0.5))
			switch {
			case names[i][w]:
				out[i] += 2 * idf
			case descs[i][w]:
				out[i] += idf
			}
		}
	}
	return out
}

func lastUser(msgs []core.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == core.RoleUser {
			return msgs[i].Content
		}
	}
	return ""
}

// forcedTool returns the function named by a tool_choice, if the caller forced one.
func forcedTool(extra map[string]json.RawMessage) string {
	raw, ok := extra["tool_choice"]
	if !ok {
		return ""
	}
	var tc struct {
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &tc) != nil {
		return ""
	}
	if tc.Function.Name != "" {
		return tc.Function.Name
	}
	return tc.Name
}

func toolChars(tools []core.ToolSpec) int {
	n := 0
	for _, t := range tools {
		n += len(t.Name) + len(t.Description) + len(t.Schema)
	}
	return n
}

func tokens(s string) map[string]bool { return textmatch.Tokens(s) }
