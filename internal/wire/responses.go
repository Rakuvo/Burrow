package wire

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/rakuvo/burrow/core"
)

// This file translates OpenAI's Responses API (POST /v1/responses) to and from
// the engine's types, for stateless function-calling use: the shape BFCL and
// most agent loops send with store=false. Anything outside that is refused with
// a clear error instead of being silently dropped, because a quietly ignored
// field would let a benchmark or an agent behave differently from how it was
// written.

// fields that are accepted and deliberately have no effect on the upstream call.
var responsesIgnored = map[string]bool{
	"store": true, "include": true, "reasoning": true, "metadata": true, "user": true,
	"safety_identifier": true, "prompt_cache_key": true, "service_tier": true, "truncation": true,
}

type respItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Output    json.RawMessage `json:"output"`
}

type respTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// DecodeResponsesRequest parses a /v1/responses request body. The returned
// error is safe to show to the client.
func DecodeResponsesRequest(body []byte) (*core.Request, error) {
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
		req          core.Request
		input        json.RawMessage
		instructions *string
		tools        []respTool
		toolChoice   json.RawMessage
		maxOut       *int
		stream, bg   bool
	)
	for _, f := range []struct {
		key  string
		into any
	}{
		{"model", &req.Model}, {"input", &input}, {"instructions", &instructions}, {"tools", &tools},
		{"tool_choice", &toolChoice}, {"temperature", &req.Temperature}, {"max_output_tokens", &maxOut},
		{"stream", &stream}, {"background", &bg},
	} {
		if err := take(f.key, f.into); err != nil {
			return nil, err
		}
	}
	if stream {
		return nil, fmt.Errorf("streaming is not supported on /v1/responses yet; send stream=false")
	}
	if bg {
		return nil, fmt.Errorf("background responses are not supported")
	}
	extra := map[string]json.RawMessage{}
	// Two "ignored" fields carry meaning the layers and the model need: the reasoning
	// effort becomes chat's reasoning_effort, and the end-user id is what Memory scopes by.
	if raw, ok := top["reasoning"]; ok {
		var r struct {
			Effort string `json:"effort"`
		}
		if json.Unmarshal(raw, &r) == nil && r.Effort != "" {
			extra["reasoning_effort"], _ = json.Marshal(r.Effort)
		}
	}
	if raw, ok := top["user"]; ok && string(raw) != "null" {
		extra["user"] = raw
	}
	for k := range responsesIgnored {
		delete(top, k)
	}
	for _, k := range []string{"top_p", "parallel_tool_calls"} {
		if v, ok := top[k]; ok {
			extra[k] = v
			delete(top, k)
		}
	}
	if len(top) > 0 { // whatever is left is something we do not implement
		left := make([]string, 0, len(top))
		for k := range top {
			left = append(left, k)
		}
		sort.Strings(left)
		return nil, fmt.Errorf("unsupported field %q on /v1/responses", left[0])
	}

	if instructions != nil && *instructions != "" {
		req.Messages = append(req.Messages, core.Message{Role: core.RoleSystem, Content: *instructions})
	}
	if err := decodeInput(input, &req); err != nil {
		return nil, err
	}
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("field \"input\" is required and must not be empty")
	}

	for i, t := range tools {
		if t.Type != "function" {
			return nil, fmt.Errorf("tools[%d]: unsupported tool type %q (only \"function\" tools are supported)", i, t.Type)
		}
		req.Tools = append(req.Tools, core.ToolSpec{Name: t.Name, Description: t.Description, Schema: string(t.Parameters)})
	}
	if maxOut != nil {
		req.MaxTokens = *maxOut
	}
	if len(toolChoice) > 0 && string(toolChoice) != "null" {
		tc, err := convertToolChoice(toolChoice)
		if err != nil {
			return nil, err
		}
		extra["tool_choice"] = tc
	}
	if len(extra) > 0 {
		req.Extra = extra
	}
	return &req, nil
}

// convertToolChoice maps the Responses form of tool_choice to the chat form.
func convertToolChoice(raw json.RawMessage) (json.RawMessage, error) {
	var mode string
	if json.Unmarshal(raw, &mode) == nil {
		return raw, nil // "auto", "none", "required" are the same in both APIs
	}
	var forced struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &forced) == nil && forced.Type == "function" && forced.Name != "" {
		return json.Marshal(map[string]any{"type": "function", "function": map[string]string{"name": forced.Name}})
	}
	return nil, fmt.Errorf("unsupported tool_choice; use \"auto\", \"none\", \"required\" or a function")
}

func decodeInput(raw json.RawMessage, req *core.Request) error {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return nil
	}
	if s[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return fmt.Errorf("field \"input\" is invalid: %v", err)
		}
		req.Messages = append(req.Messages, core.Message{Role: core.RoleUser, Content: text})
		return nil
	}
	var items []respItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return fmt.Errorf("field \"input\" must be a string or a list of items: %v", err)
	}
	for i, it := range items {
		switch {
		case it.Type == "function_call":
			call := core.ToolCall{ID: it.CallID, Name: it.Name, Arguments: it.Arguments}
			// Calls that follow an assistant message belong to that same turn.
			if n := len(req.Messages); n > 0 && req.Messages[n-1].Role == core.RoleAssistant && req.Messages[n-1].ToolCallID == "" {
				req.Messages[n-1].ToolCalls = append(req.Messages[n-1].ToolCalls, call)
			} else {
				req.Messages = append(req.Messages, core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{call}})
			}
		case it.Type == "function_call_output":
			out, err := outputText(it.Output)
			if err != nil {
				return fmt.Errorf("input[%d]: %v", i, err)
			}
			req.Messages = append(req.Messages, core.Message{Role: core.RoleTool, ToolCallID: it.CallID, Content: out})
		case it.Type == "reasoning":
			// A chat-completions upstream cannot take reasoning items back, so
			// they are dropped. Reasoning models lose their earlier thinking here.
		case it.Type == "" || it.Type == "message":
			text, err := contentTextResponses(it.Content)
			if err != nil {
				return fmt.Errorf("input[%d]: %v", i, err)
			}
			role := it.Role
			if role == "developer" {
				role = "system" // upstreams that predate the developer role only know system
			}
			if role == "" {
				return fmt.Errorf("input[%d]: message has no role", i)
			}
			req.Messages = append(req.Messages, core.Message{Role: core.Role(role), Content: text})
		default:
			return fmt.Errorf("input[%d]: unsupported item type %q", i, it.Type)
		}
	}
	return nil
}

func outputText(raw json.RawMessage) (string, error) {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return "", nil
	}
	if s[0] == '"' {
		var out string
		err := json.Unmarshal(raw, &out)
		return out, err
	}
	return contentTextResponses(raw)
}

// contentTextResponses flattens message content: a string, or a list of text parts.
func contentTextResponses(raw json.RawMessage) (string, error) {
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
			Type    string `json:"type"`
			Text    string `json:"text"`
			Refusal string `json:"refusal"`
		}
		if err := json.Unmarshal(raw, &parts); err != nil {
			return "", err
		}
		var b strings.Builder
		for _, p := range parts {
			switch p.Type {
			case "input_text", "output_text", "text":
				b.WriteString(p.Text)
			case "refusal":
				b.WriteString(p.Refusal)
			default:
				return "", fmt.Errorf("unsupported content part type %q (only text is supported)", p.Type)
			}
		}
		return b.String(), nil
	}
	return "", fmt.Errorf("content must be a string or a list of parts")
}

// EncodeResponses renders r as a /v1/responses body. id and created identify
// the response; item ids are derived from id so they stay stable and unique.
func EncodeResponses(r *core.Response, id string, created int64) ([]byte, error) {
	var output []any
	if r.Message.Content != "" || len(r.Message.ToolCalls) == 0 {
		output = append(output, map[string]any{
			"id": "msg_" + id, "type": "message", "status": "completed", "role": "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": r.Message.Content, "annotations": []any{}}},
		})
	}
	for i, tc := range r.Message.ToolCalls {
		output = append(output, map[string]any{
			"id": fmt.Sprintf("fc_%s_%d", id, i), "type": "function_call", "status": "completed",
			"call_id": tc.ID, "name": tc.Name, "arguments": tc.Arguments,
		})
	}

	status, incomplete := "completed", any(nil)
	switch r.FinishReason {
	case "length":
		status, incomplete = "incomplete", map[string]string{"reason": "max_output_tokens"}
	case "content_filter":
		status, incomplete = "incomplete", map[string]string{"reason": "content_filter"}
	}
	u := r.Usage
	return json.Marshal(map[string]any{
		"id": "resp_" + id, "object": "response", "created_at": created, "status": status,
		"error": nil, "incomplete_details": incomplete, "instructions": nil, "metadata": map[string]any{},
		"model": r.Model, "output": output, "parallel_tool_calls": true,
		"temperature": nil, "tool_choice": "auto", "tools": []any{}, "top_p": nil,
		"usage": map[string]any{
			"input_tokens": u.InputTokens, "output_tokens": u.OutputTokens, "total_tokens": u.InputTokens + u.OutputTokens,
			"input_tokens_details":  map[string]int{"cached_tokens": u.CachedTokens},
			"output_tokens_details": map[string]int{"reasoning_tokens": 0},
		},
	})
}
