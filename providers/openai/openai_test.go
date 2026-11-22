package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/providers/openai"
)

func serve(t *testing.T, status int, body string, hdr map[string]string, got *map[string]any, auth *string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if auth != nil {
			*auth = r.Header.Get("Authorization")
		}
		if got != nil {
			b, _ := io.ReadAll(r.Body)
			if err := json.Unmarshal(b, got); err != nil {
				t.Errorf("bad request json: %v", err)
			}
		}
		for k, v := range hdr {
			w.Header().Set(k, v)
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestCompleteTextAndUsage(t *testing.T) {
	var sent map[string]any
	var auth string
	srv := serve(t, 200, `{"model":"gpt-x","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":12,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":8}}}`, nil, &sent, &auth)

	p := openai.New(openai.Config{BaseURL: srv.URL + "/", APIKey: "k"})
	temp := 0.2
	resp, err := p.Complete(context.Background(), core.Request{
		Model: "gpt-x", Temperature: &temp, MaxTokens: 50,
		Messages: []core.Message{{Role: core.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "hello" || resp.FinishReason != "stop" || resp.Model != "gpt-x" {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Usage != (core.Usage{InputTokens: 12, OutputTokens: 3, CachedTokens: 8}) {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if auth != "Bearer k" {
		t.Fatalf("auth = %q", auth)
	}
	if sent["max_tokens"] != float64(50) || sent["temperature"] != 0.2 {
		t.Fatalf("request = %v", sent)
	}
}

func TestMaxTokensFieldAndOmitTemperature(t *testing.T) {
	var sent map[string]any
	srv := serve(t, 200, `{"choices":[{"message":{"content":"x"}}]}`, nil, &sent, nil)
	p := openai.New(openai.Config{BaseURL: srv.URL, MaxTokensField: openai.MaxTokensCompletion, OmitTemperature: true})
	temp := 1.0
	_, err := p.Complete(context.Background(), core.Request{Model: "m", MaxTokens: 9, Temperature: &temp})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := sent["max_tokens"]; ok {
		t.Fatal("max_tokens must not be sent")
	}
	if sent["max_completion_tokens"] != float64(9) {
		t.Fatalf("request = %v", sent)
	}
	if _, ok := sent["temperature"]; ok {
		t.Fatal("temperature must be omitted")
	}
}

func TestToolsAndToolCallRoundTrip(t *testing.T) {
	var sent map[string]any
	srv := serve(t, 200, `{"choices":[{"message":{"role":"assistant","content":null,
		"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_w","arguments":"{\"city\":\"Pune\"}"}}]},
		"finish_reason":"tool_calls"}]}`, nil, &sent, nil)
	p := openai.New(openai.Config{BaseURL: srv.URL})

	resp, err := p.Complete(context.Background(), core.Request{
		Model: "m",
		Tools: []core.ToolSpec{{Name: "get_w", Description: "weather", Schema: `{"type":"object","properties":{"city":{"type":"string"}}}`}},
		Messages: []core.Message{
			{Role: core.RoleUser, Content: "weather?"},
			{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "c0", Name: "get_w", Arguments: `{"city":"Goa"}`}}},
			{Role: core.RoleTool, ToolCallID: "c0", Content: "sunny"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0] != (core.ToolCall{ID: "c1", Name: "get_w", Arguments: `{"city":"Pune"}`}) {
		t.Fatalf("tool calls = %+v", resp.Message.ToolCalls)
	}
	if resp.Message.Content != "" || resp.FinishReason != "tool_calls" {
		t.Fatalf("resp = %+v", resp)
	}

	msgs := sent["messages"].([]any)
	asst := msgs[1].(map[string]any)
	if asst["content"] != nil {
		t.Fatalf("tool-call-only assistant turn must have null content, got %v", asst["content"])
	}
	if msgs[2].(map[string]any)["tool_call_id"] != "c0" {
		t.Fatalf("tool message = %v", msgs[2])
	}
	tool := sent["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if tool["name"] != "get_w" || tool["parameters"].(map[string]any)["type"] != "object" {
		t.Fatalf("tool = %v", tool)
	}
}

func TestInvalidToolSchemaFailsBeforeNetwork(t *testing.T) {
	p := openai.New(openai.Config{BaseURL: "http://127.0.0.1:1"}) // would fail to connect if reached
	_, err := p.Complete(context.Background(), core.Request{Tools: []core.ToolSpec{{Name: "bad", Schema: "{nope"}}})
	if err == nil || err.Error() != `openai: tool "bad" has an invalid JSON schema` {
		t.Fatalf("err = %v", err)
	}
}

func TestAPIError(t *testing.T) {
	srv := serve(t, 429, `{"error":{"message":"slow down","type":"rate_limit","code":"rl"}}`,
		map[string]string{"Retry-After": "7"}, nil, nil)
	p := openai.New(openai.Config{BaseURL: srv.URL})
	_, err := p.Complete(context.Background(), core.Request{Model: "m"})
	var ae *openai.APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %v, want *APIError", err)
	}
	if ae.Status != 429 || ae.Message != "slow down" || ae.Code != "rl" || ae.RetryAfter != 7*time.Second || !ae.Retryable() {
		t.Fatalf("api error = %+v", ae)
	}
}

func TestNonJSONErrorBodyAndNonRetryable(t *testing.T) {
	srv := serve(t, 401, "unauthorized", nil, nil, nil)
	p := openai.New(openai.Config{BaseURL: srv.URL})
	_, err := p.Complete(context.Background(), core.Request{})
	var ae *openai.APIError
	if !errors.As(err, &ae) || ae.Message != "unauthorized" || ae.Retryable() {
		t.Fatalf("err = %v", err)
	}
}

func TestNoChoicesIsAnError(t *testing.T) {
	srv := serve(t, 200, `{"choices":[]}`, nil, nil, nil)
	p := openai.New(openai.Config{BaseURL: srv.URL})
	if _, err := p.Complete(context.Background(), core.Request{}); err == nil {
		t.Fatal("want error")
	}
}

func TestContextCancellation(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release) // runs first, so Close never waits on a parked handler
	p := openai.New(openai.Config{BaseURL: srv.URL})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := p.Complete(ctx, core.Request{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
}

// Compile-time check that Provider satisfies the engine's interface.
var _ core.Provider = (*openai.Provider)(nil)

func TestExtraFieldsAreForwardedButNeverStream(t *testing.T) {
	var sent map[string]any
	srv := serve(t, 200, `{"choices":[{"message":{"content":"x"}}]}`, nil, &sent, nil)
	p := openai.New(openai.Config{BaseURL: srv.URL})
	_, err := p.Complete(context.Background(), core.Request{
		Model: "m",
		Extra: map[string]json.RawMessage{
			"tool_choice":    json.RawMessage(`"required"`),
			"stream":         json.RawMessage(`true`),
			"stream_options": json.RawMessage(`{"include_usage":true}`),
			"model":          json.RawMessage(`"hijack"`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if sent["tool_choice"] != "required" {
		t.Fatalf("tool_choice not forwarded: %v", sent)
	}
	if _, ok := sent["stream"]; ok {
		t.Fatal("stream must never be forwarded; the engine needs whole responses")
	}
	if _, ok := sent["stream_options"]; ok {
		t.Fatal("stream_options must not be forwarded")
	}
	if sent["model"] != "m" {
		t.Fatalf("Extra must not override modelled fields: %v", sent["model"])
	}
}
