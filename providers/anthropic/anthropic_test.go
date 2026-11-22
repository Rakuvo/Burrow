package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/rakuvo/burrow/core"
)

func TestEncodeMapsRolesToolsAndChoice(t *testing.T) {
	req := core.Request{
		Model: "claude-x",
		Messages: []core.Message{
			{Role: core.RoleSystem, Content: "be brief"},
			{Role: core.RoleUser, Content: "refund A-1"},
			{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "c1", Name: "issue_refund", Arguments: `{"order_id":"A-1"}`}, {ID: "c2", Name: "get_order", Arguments: "oops"}}},
			{Role: core.RoleTool, ToolCallID: "c1", Content: "ok"},
			{Role: core.RoleTool, ToolCallID: "c2", Content: "{}"},
		},
		Tools: []core.ToolSpec{{Name: "issue_refund", Description: "d", Schema: `{"type":"object"}`}},
		Extra: map[string]json.RawMessage{"tool_choice": json.RawMessage(`"required"`)},
	}
	raw, err := EncodeRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	var b struct {
		System     string `json:"system"`
		MaxTokens  int    `json:"max_tokens"`
		ToolChoice struct {
			Type string `json:"type"`
		} `json:"tool_choice"`
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type  string          `json:"type"`
				Input json.RawMessage `json:"input"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &b); err != nil {
		t.Fatal(err)
	}
	if b.System != "be brief" || b.MaxTokens != defaultMaxTokens || b.ToolChoice.Type != "any" {
		t.Fatalf("system/max_tokens/tool_choice wrong: %s", raw)
	}
	if len(b.Messages) != 3 || b.Messages[2].Role != "user" || len(b.Messages[2].Content) != 2 {
		t.Fatalf("both tool results must merge into one user turn: %s", raw)
	}
	if string(b.Messages[1].Content[1].Input) != "{}" {
		t.Fatalf("invalid tool arguments must become an empty object, got %s", b.Messages[1].Content[1].Input)
	}
}

func TestCompleteRoundTrip(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "k" || r.Header.Get("anthropic-version") == "" || r.URL.Path != "/v1/messages" {
			t.Errorf("bad headers or path: %v %s", r.Header, r.URL.Path)
		}
		io.Copy(io.Discard, r.Body)
		io.WriteString(w, `{"model":"claude-x","stop_reason":"tool_use","content":[{"type":"text","text":"on it"},{"type":"tool_use","id":"t1","name":"get_order","input":{"order_id":"A-1"}}],"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":4}}`)
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL + "/v1", APIKey: "k"})
	resp, err := p.Complete(context.Background(), core.Request{Model: "claude-x", Messages: []core.Message{{Role: core.RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "tool_calls" || len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Arguments != `{"order_id":"A-1"}` || resp.Message.Content != "on it" {
		t.Fatalf("bad decode: %+v", resp)
	}
	if resp.Usage.InputTokens != 14 || resp.Usage.CachedTokens != 4 || resp.Usage.OutputTokens != 5 {
		t.Fatalf("bad usage: %+v", resp.Usage)
	}
}

func TestErrorIsStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(429)
		io.WriteString(w, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
	}))
	defer srv.Close()
	_, err := New(Config{BaseURL: srv.URL}).Complete(context.Background(), core.Request{Messages: []core.Message{{Role: core.RoleUser, Content: "x"}}})
	se, ok := err.(core.StatusError)
	if !ok || se.StatusCode() != 429 || se.RetryAfterDuration().Seconds() != 7 {
		t.Fatalf("want a 429 StatusError with Retry-After 7, got %v", err)
	}
}
