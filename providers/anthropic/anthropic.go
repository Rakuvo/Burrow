// Package anthropic is a core.Provider for Claude's native Messages API.
// The gateway still speaks the OpenAI format to the agent; this adapter
// translates each call, so an agent written for OpenAI can run on Claude.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/rakuvo/burrow/core"
)

const (
	DefaultBaseURL   = "https://api.anthropic.com/v1"
	apiVersion       = "2023-06-01"
	defaultMaxTokens = 4096
	maxBodyBytes     = 32 << 20
)

type Config struct {
	Name       string // defaults to "anthropic"
	BaseURL    string // defaults to DefaultBaseURL
	APIKey     string
	HTTPClient *http.Client
}

type Provider struct{ cfg Config }

func New(cfg Config) *Provider {
	if cfg.Name == "" {
		cfg.Name = "anthropic"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 5 * time.Minute}
	}
	return &Provider{cfg: cfg}
}

func (p *Provider) Name() string { return p.cfg.Name }

// APIError is a non-2xx answer. It implements core.StatusError.
type APIError struct {
	Status     int
	Type       string
	Message    string
	RetryAfter time.Duration
}

func (e *APIError) Error() string                     { return fmt.Sprintf("anthropic: HTTP %d: %s", e.Status, e.Message) }
func (e *APIError) StatusCode() int                   { return e.Status }
func (e *APIError) RetryAfterDuration() time.Duration { return e.RetryAfter }
func (e *APIError) Retryable() bool {
	return e.Status == 429 || e.Status == 529 || e.Status >= 500
}

type block struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
}

type msg struct {
	Role    string  `json:"role"`
	Content []block `json:"content"`
}

// EncodeRequest builds the Messages API body for a neutral request.
func EncodeRequest(req core.Request) ([]byte, error) {
	var system []string
	var msgs []msg
	add := func(role string, bs ...block) {
		if n := len(msgs); n > 0 && msgs[n-1].Role == role {
			msgs[n-1].Content = append(msgs[n-1].Content, bs...)
			return
		}
		msgs = append(msgs, msg{Role: role, Content: bs})
	}
	for _, m := range req.Messages {
		switch m.Role {
		case core.RoleSystem:
			system = append(system, m.Content)
		case core.RoleUser:
			add("user", block{Type: "text", Text: nonEmpty(m.Content)})
		case core.RoleAssistant:
			var bs []block
			if m.Content != "" {
				bs = append(bs, block{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				bs = append(bs, block{Type: "tool_use", ID: tc.ID, Name: tc.Name, Input: object(tc.Arguments)})
			}
			if len(bs) == 0 {
				bs = []block{{Type: "text", Text: "(no content)"}}
			}
			add("assistant", bs...)
		case core.RoleTool:
			add("user", block{Type: "tool_result", ToolUseID: m.ToolCallID, Content: nonEmpty(m.Content)})
		}
	}
	if len(msgs) == 0 || msgs[0].Role != "user" {
		msgs = append([]msg{{Role: "user", Content: []block{{Type: "text", Text: "(start)"}}}}, msgs...)
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	body := map[string]any{"model": req.Model, "max_tokens": maxTokens, "messages": msgs}
	if len(system) > 0 {
		body["system"] = strings.Join(system, "\n\n")
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		var ts []map[string]any
		for _, t := range req.Tools {
			schema := json.RawMessage(t.Schema)
			if !json.Valid(schema) || len(bytes.TrimSpace(schema)) == 0 {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			ts = append(ts, map[string]any{"name": t.Name, "description": t.Description, "input_schema": schema})
		}
		body["tools"] = ts
		if tc, ok := toolChoice(req.Extra["tool_choice"]); ok {
			body["tool_choice"] = tc
		}
	}
	if raw, ok := req.Extra["stop"]; ok {
		var one string
		var many []string
		if json.Unmarshal(raw, &many) == nil {
			body["stop_sequences"] = many
		} else if json.Unmarshal(raw, &one) == nil && one != "" {
			body["stop_sequences"] = []string{one}
		}
	}
	if raw, ok := req.Extra["top_p"]; ok && req.Temperature == nil {
		body["top_p"] = raw
	}
	return json.Marshal(body)
}

// toolChoice maps the OpenAI tool_choice values to Claude's.
func toolChoice(raw json.RawMessage) (any, bool) {
	if len(raw) == 0 {
		return nil, false
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "auto":
			return map[string]string{"type": "auto"}, true
		case "required":
			return map[string]string{"type": "any"}, true
		case "none":
			return map[string]string{"type": "none"}, true
		}
		return nil, false
	}
	var o struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &o) == nil && o.Function.Name != "" {
		return map[string]string{"type": "tool", "name": o.Function.Name}, true
	}
	return nil, false
}

func nonEmpty(s string) string {
	if s == "" {
		return "(empty)"
	}
	return s
}

// object returns s if it is a JSON object, else {} (Claude requires an object).
func object(s string) json.RawMessage {
	s = strings.TrimSpace(s)
	var m map[string]json.RawMessage
	if s != "" && json.Unmarshal([]byte(s), &m) == nil {
		return json.RawMessage(s)
	}
	return json.RawMessage("{}")
}

// DecodeResponse reads a Messages API answer into the neutral form.
func DecodeResponse(raw []byte) (*core.Response, error) {
	var w struct {
		Model      string  `json:"model"`
		StopReason string  `json:"stop_reason"`
		Content    []block `json:"content"`
		Usage      struct {
			In         int `json:"input_tokens"`
			Out        int `json:"output_tokens"`
			CacheRead  int `json:"cache_read_input_tokens"`
			CacheWrite int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("anthropic: decoding response: %w", err)
	}
	m := core.Message{Role: core.RoleAssistant}
	var texts []string
	for _, b := range w.Content {
		switch b.Type {
		case "text":
			texts = append(texts, b.Text)
		case "tool_use":
			args := strings.TrimSpace(string(b.Input))
			if args == "" {
				args = "{}"
			}
			m.ToolCalls = append(m.ToolCalls, core.ToolCall{ID: b.ID, Name: b.Name, Arguments: args})
		}
	}
	m.Content = strings.Join(texts, "")
	finish := map[string]string{"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length"}[w.StopReason]
	if finish == "" {
		finish = "stop"
	}
	return &core.Response{
		Model: w.Model, Message: m, FinishReason: finish,
		Usage: core.Usage{
			InputTokens:  w.Usage.In + w.Usage.CacheRead + w.Usage.CacheWrite,
			OutputTokens: w.Usage.Out,
			CachedTokens: w.Usage.CacheRead,
		},
	}, nil
}

func (p *Provider) Complete(ctx context.Context, req core.Request) (*core.Response, error) {
	body, err := EncodeRequest(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hr.Header.Set("Content-Type", "application/json")
	hr.Header.Set("anthropic-version", apiVersion)
	if p.cfg.APIKey != "" {
		hr.Header.Set("x-api-key", p.cfg.APIKey)
	}
	res, err := p.cfg.HTTPClient.Do(hr)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, maxBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("anthropic: reading response: %w", err)
	}
	if res.StatusCode < 200 || res.StatusCode > 299 {
		return nil, parseError(res, raw)
	}
	return DecodeResponse(raw)
}

func parseError(res *http.Response, raw []byte) *APIError {
	e := &APIError{Status: res.StatusCode}
	var w struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &w) == nil && w.Error.Message != "" {
		e.Message, e.Type = w.Error.Message, w.Error.Type
	} else {
		e.Message = strings.TrimSpace(string(raw))
		if len(e.Message) > 300 {
			e.Message = e.Message[:300]
		}
		if e.Message == "" {
			e.Message = http.StatusText(res.StatusCode)
		}
	}
	if s, err := strconv.Atoi(res.Header.Get("Retry-After")); err == nil && s >= 0 {
		e.RetryAfter = time.Duration(s) * time.Second
	}
	return e
}
