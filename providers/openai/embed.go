package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/rakuvo/burrow/core"
)

// Embed returns one embedding vector per input text, in order, using the endpoint's
// /embeddings route. Inputs are sent in batches, because the API caps how many can go in
// one request. The usage reports input tokens only.
// maxEmbedChars keeps one input under the embedding models' 8,192-token limit even for
// token-dense text such as code (a rough 2 characters per token in the worst case). Longer
// texts are cut to their start: the vector then describes the opening, and the request does
// not fail for every other text in the batch.
const maxEmbedChars = 12000

func (p *Provider) Embed(ctx context.Context, model string, texts []string) ([][]float32, core.Usage, error) {
	const batch = 128
	texts = capLengths(texts)
	var usage core.Usage
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += batch {
		end := min(start+batch, len(texts))
		body, err := json.Marshal(map[string]any{"model": model, "input": texts[start:end]})
		if err != nil {
			return nil, usage, err
		}
		hr, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/embeddings", bytes.NewReader(body))
		if err != nil {
			return nil, usage, err
		}
		hr.Header.Set("Content-Type", "application/json")
		if p.cfg.APIKey != "" {
			hr.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
		}
		res, err := p.cfg.HTTPClient.Do(hr)
		if err != nil {
			return nil, usage, fmt.Errorf("openai: %w", err)
		}
		raw, err := io.ReadAll(io.LimitReader(res.Body, maxBodyBytes))
		res.Body.Close()
		if err != nil {
			return nil, usage, fmt.Errorf("openai: reading embeddings: %w", err)
		}
		if res.StatusCode < 200 || res.StatusCode > 299 {
			return nil, usage, parseAPIError(res, raw)
		}
		var w struct {
			Data []struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(raw, &w); err != nil {
			return nil, usage, fmt.Errorf("openai: bad embeddings response: %w", err)
		}
		if len(w.Data) != end-start {
			return nil, usage, fmt.Errorf("openai: asked for %d embeddings, got %d", end-start, len(w.Data))
		}
		vecs := make([][]float32, len(w.Data))
		for _, d := range w.Data {
			if d.Index < 0 || d.Index >= len(vecs) {
				return nil, usage, fmt.Errorf("openai: embedding index %d out of range", d.Index)
			}
			vecs[d.Index] = d.Embedding
		}
		out = append(out, vecs...)
		usage.InputTokens += w.Usage.PromptTokens
	}
	return out, usage, nil
}

// cutUTF8 shortens s to at most n bytes without splitting a character.
func cutUTF8(s string, n int) string {
	for n > 0 && n < len(s) && s[n]&0xC0 == 0x80 {
		n--
	}
	return s[:n]
}

// capLengths returns texts with every over-long entry cut; the caller's slice is not changed.
func capLengths(texts []string) []string {
	var out []string
	for i, t := range texts {
		if len(t) <= maxEmbedChars {
			continue
		}
		if out == nil {
			out = append([]string(nil), texts...)
		}
		out[i] = cutUTF8(t, maxEmbedChars)
	}
	if out == nil {
		return texts
	}
	return out
}
