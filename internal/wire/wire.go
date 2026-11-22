// Package wire converts between core types and the OpenAI chat-completions JSON
// format. The openai provider (client side) and the proxy (server side) share it,
// so the two directions cannot drift apart.
package wire

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/rakuvo/burrow/core"
)

// Field names used for the output token limit.
const (
	MaxTokensLegacy     = "max_tokens"
	MaxTokensCompletion = "max_completion_tokens"
)

const emptySchema = `{"type":"object","properties":{}}`

type toolCall struct {
	Index    *int     `json:"index,omitempty"` // streaming deltas only
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function function `json:"function"`
}

type function struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ---------------------------------------------------------------- requests

// RequestOptions tunes how a core.Request is written for a particular server.
type RequestOptions struct {
	MaxTokensField  string // defaults to MaxTokensLegacy
	OmitTemperature bool
}

type outMessage struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type outTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

// EncodeRequest renders req as a chat-completions request body. It never asks
// for streaming: the engine always works on complete responses.
func EncodeRequest(req core.Request, o RequestOptions) ([]byte, error) {
	if o.MaxTokensField == "" {
		o.MaxTokensField = MaxTokensLegacy
	}
	m := map[string]any{"model": req.Model}

	msgs := make([]outMessage, 0, len(req.Messages))
	for _, cm := range req.Messages {
		wm := outMessage{Role: string(cm.Role), ToolCallID: cm.ToolCallID}
		// An assistant turn that only carries tool calls has null content.
		if cm.Content != "" || len(cm.ToolCalls) == 0 {
			c := cm.Content
			wm.Content = &c
		}
		for _, tc := range cm.ToolCalls {
			wm.ToolCalls = append(wm.ToolCalls, toolCall{
				ID: tc.ID, Type: "function",
				Function: function{Name: tc.Name, Arguments: tc.Arguments},
			})
		}
		msgs = append(msgs, wm)
	}
	m["messages"] = msgs

	if len(req.Tools) > 0 {
		tools := make([]outTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			schema := t.Schema
			if schema == "" {
				schema = emptySchema
			}
			if !json.Valid([]byte(schema)) {
				return nil, fmt.Errorf("tool %q has an invalid JSON schema", t.Name)
			}
			wt := outTool{Type: "function"}
			wt.Function.Name = t.Name
			wt.Function.Description = t.Description
			wt.Function.Parameters = json.RawMessage(schema)
			tools = append(tools, wt)
		}
		m["tools"] = tools
	}
	if req.Temperature != nil && !o.OmitTemperature {
		m["temperature"] = *req.Temperature
	}
	if req.MaxTokens > 0 {
		m[o.MaxTokensField] = req.MaxTokens
	}
	for k, v := range req.Extra {
		if _, taken := m[k]; taken || k == "stream" || k == "stream_options" {
			continue
		}
		m[k] = v
	}
	return json.Marshal(m)
}

// Parsed is a decoded client request plus the streaming flags the engine does
// not carry itself.
type Parsed struct {
	Request      core.Request
	Stream       bool
	IncludeUsage bool
}

type inMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []toolCall      `json:"tool_calls"`
	ToolCallID string          `json:"tool_call_id"`
}

type inTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

// DecodeRequest parses a chat-completions request body. Fields the engine does
// not model land in Request.Extra instead of being dropped. The returned error
// is safe to show to the client.
func DecodeRequest(body []byte) (*Parsed, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, fmt.Errorf("request body is not valid JSON: %v", err)
	}
	take := func(key string, into any) error {
		raw, ok := top[key]
		delete(top, key)
		if !ok {
			return nil
		}
		if err := json.Unmarshal(raw, into); err != nil {
			return fmt.Errorf("field %q is invalid: %v", key, err)
		}
		return nil
	}

	var (
		p       Parsed
		msgs    []inMessage
		tools   []inTool
		maxTok  *int
		maxComp *int
		opts    struct {
			IncludeUsage bool `json:"include_usage"`
		}
	)
	for _, f := range []struct {
		key  string
		into any
	}{
		{"model", &p.Request.Model},
		{"messages", &msgs},
		{"tools", &tools},
		{"temperature", &p.Request.Temperature},
		{"max_tokens", &maxTok},
		{"max_completion_tokens", &maxComp},
		{"stream", &p.Stream},
		{"stream_options", &opts},
	} {
		if err := take(f.key, f.into); err != nil {
			return nil, err
		}
	}
	p.IncludeUsage = opts.IncludeUsage

	if len(msgs) == 0 {
		return nil, fmt.Errorf("field \"messages\" is required and must not be empty")
	}
	for i, m := range msgs {
		text, err := contentText(m.Content)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %v", i, err)
		}
		cm := core.Message{Role: core.Role(m.Role), Content: text, ToolCallID: m.ToolCallID}
		for _, tc := range m.ToolCalls {
			cm.ToolCalls = append(cm.ToolCalls, core.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
		p.Request.Messages = append(p.Request.Messages, cm)
	}
	for i, t := range tools {
		if t.Type != "function" {
			return nil, fmt.Errorf("tools[%d]: unsupported tool type %q (only \"function\" tools are supported)", i, t.Type)
		}
		p.Request.Tools = append(p.Request.Tools, core.ToolSpec{
			Name: t.Function.Name, Description: t.Function.Description, Schema: string(t.Function.Parameters),
		})
	}
	switch {
	case maxComp != nil:
		p.Request.MaxTokens = *maxComp
	case maxTok != nil:
		p.Request.MaxTokens = *maxTok
	}
	if len(top) > 0 {
		p.Request.Extra = top
	}
	return &p, nil
}

// contentText flattens message content, which is either a string or a list of
// typed parts. Only text parts are supported.
func contentText(raw json.RawMessage) (string, error) {
	s := strings.TrimSpace(string(raw))
	switch {
	case s == "" || s == "null":
		return "", nil
	case s[0] == '"':
		var out string
		err := json.Unmarshal(raw, &out)
		return out, err
	case s[0] == '[':
		var parts []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(raw, &parts); err != nil {
			return "", err
		}
		var b strings.Builder
		for _, p := range parts {
			if p.Type != "text" {
				return "", fmt.Errorf("unsupported content part type %q (only \"text\" is supported)", p.Type)
			}
			b.WriteString(p.Text)
		}
		return b.String(), nil
	}
	return "", fmt.Errorf("content must be a string or a list of parts")
}

// --------------------------------------------------------------- responses

type usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func newUsage(u core.Usage) *usage {
	w := &usage{PromptTokens: u.InputTokens, CompletionTokens: u.OutputTokens, TotalTokens: u.InputTokens + u.OutputTokens}
	w.PromptTokensDetails.CachedTokens = u.CachedTokens
	return w
}

type inResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content   *string    `json:"content"`
			ToolCalls []toolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

// DecodeResponse parses a chat-completions response body.
func DecodeResponse(raw []byte) (*core.Response, error) {
	var w inResponse
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	if len(w.Choices) == 0 {
		return nil, fmt.Errorf("response has no choices")
	}
	c := w.Choices[0]
	msg := core.Message{Role: core.RoleAssistant}
	if c.Message.Content != nil {
		msg.Content = *c.Message.Content
	}
	for _, tc := range c.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, core.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
	}
	return &core.Response{
		Model:        w.Model,
		Message:      msg,
		FinishReason: c.FinishReason,
		Usage: core.Usage{
			InputTokens:  w.Usage.PromptTokens,
			OutputTokens: w.Usage.CompletionTokens,
			CachedTokens: w.Usage.PromptTokensDetails.CachedTokens,
		},
	}, nil
}

func finishReason(r *core.Response) string {
	if r.FinishReason == "" {
		return "stop"
	}
	return r.FinishReason
}

func outToolCalls(r *core.Response, withIndex bool) []toolCall {
	var out []toolCall
	for i, tc := range r.Message.ToolCalls {
		w := toolCall{ID: tc.ID, Type: "function", Function: function{Name: tc.Name, Arguments: tc.Arguments}}
		if withIndex {
			idx := i
			w.Index = &idx
		}
		out = append(out, w)
	}
	return out
}

// EncodeResponse renders r as a chat.completion body.
func EncodeResponse(r *core.Response, id string, created int64) ([]byte, error) {
	m := outMessage{Role: "assistant", ToolCalls: outToolCalls(r, false)}
	if r.Message.Content != "" || len(m.ToolCalls) == 0 {
		c := r.Message.Content
		m.Content = &c
	}
	return json.Marshal(map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   r.Model,
		"choices": []any{map[string]any{"index": 0, "message": m, "finish_reason": finishReason(r)}},
		"usage":   newUsage(r.Usage),
	})
}

// StreamChunks renders r as the data payloads of a chat.completion.chunk stream
// (without the "data: " framing or the final [DONE]). The whole response already
// exists by the time we stream it, because layers must validate it first.
func StreamChunks(r *core.Response, id string, created int64, includeUsage bool) ([][]byte, error) {
	chunk := func(delta any, finish *string, withUsage bool) ([]byte, error) {
		m := map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": r.Model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
		}
		if withUsage {
			m["usage"] = newUsage(r.Usage)
		}
		return json.Marshal(m)
	}

	delta := map[string]any{"role": "assistant"}
	if r.Message.Content != "" {
		delta["content"] = r.Message.Content
	}
	if tcs := outToolCalls(r, true); len(tcs) > 0 {
		delta["tool_calls"] = tcs
	}
	first, err := chunk(delta, nil, false)
	if err != nil {
		return nil, err
	}
	fr := finishReason(r)
	last, err := chunk(map[string]any{}, &fr, false)
	if err != nil {
		return nil, err
	}
	out := [][]byte{first, last}

	if includeUsage {
		// Per the API, the usage chunk has an empty choices list.
		u, err := json.Marshal(map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": r.Model,
			"choices": []any{}, "usage": newUsage(r.Usage),
		})
		if err != nil {
			return nil, err
		}
		out = append(out, u)
	}
	return out, nil
}
