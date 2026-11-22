// Package core is the Rakuvo engine: the step pipeline every model call passes through.
package core

import "encoding/json"

// Role is the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall is a model's request to invoke a tool. Arguments is raw JSON so that
// layers can validate and repair it before anything parses it.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// ToolSpec describes a tool the model may call. Schema is a JSON Schema document.
type ToolSpec struct {
	Name        string
	Description string
	Schema      string
}

// Message is one turn of the conversation.
type Message struct {
	Role       Role
	Content    string
	ToolCalls  []ToolCall
	ToolCallID string // set when Role == RoleTool
}

// Request is a provider-neutral model request.
type Request struct {
	Model       string
	Messages    []Message
	Tools       []ToolSpec
	Temperature *float64
	MaxTokens   int

	// Extra holds request fields the engine does not model (tool_choice,
	// response_format, stop, seed, ...). Providers forward them verbatim so a
	// proxied agent behaves as it would against the model directly.
	Extra map[string]json.RawMessage
}

// Usage is token accounting for one provider call.
type Usage struct {
	InputTokens  int
	OutputTokens int
	CachedTokens int
}

// Response is a provider-neutral model response.
type Response struct {
	Model        string
	Message      Message
	FinishReason string
	Usage        Usage
}

// Clone returns a copy of r that shares no slices with the original, so a layer
// can rewrite a request without affecting the caller's copy or a retry.
func (r Request) Clone() Request {
	c := r
	c.Messages = make([]Message, len(r.Messages))
	for i, m := range r.Messages {
		m.ToolCalls = append([]ToolCall(nil), m.ToolCalls...)
		c.Messages[i] = m
	}
	c.Tools = append([]ToolSpec(nil), r.Tools...)
	if r.Extra != nil {
		c.Extra = make(map[string]json.RawMessage, len(r.Extra))
		for k, v := range r.Extra {
			c.Extra[k] = v
		}
	}
	return c
}
