package proxy_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/layers/guard"
	"github.com/rakuvo/burrow/providers/mock"
	"github.com/rakuvo/burrow/proxy"
)

func newServer(t *testing.T, p core.Provider, tokens ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer((&proxy.Server{Engine: &core.Engine{Provider: p}, Tokens: tokens}).Handler())
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, url, body string, hdr ...string) (*http.Response, []byte) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	return res, b
}

func TestChatRoundTripWithToolCallAndExtraPassthrough(t *testing.T) {
	p := mock.New(mock.Reply{Response: &core.Response{
		Model: "m1",
		Message: core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{
			{ID: "c1", Name: "lookup", Arguments: `{"id":7}`},
		}},
		FinishReason: "tool_calls",
		Usage:        core.Usage{InputTokens: 20, OutputTokens: 4, CachedTokens: 16},
	}})
	srv := newServer(t, p)

	res, body := post(t, srv.URL+"/v1/chat/completions", `{
		"model":"m1","temperature":0.3,"max_completion_tokens":64,"tool_choice":"required","seed":5,
		"tools":[{"type":"function","function":{"name":"lookup","description":"d","parameters":{"type":"object"}}}],
		"messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"he"},{"type":"text","text":"llo"}]}]
	}`)
	if res.StatusCode != 200 {
		t.Fatalf("status %d: %s", res.StatusCode, body)
	}

	var out struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content   *string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct{ Name, Arguments string }
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			TotalTokens         int `json:"total_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	c := out.Choices[0]
	if out.Object != "chat.completion" || !strings.HasPrefix(out.ID, "chatcmpl-") ||
		c.FinishReason != "tool_calls" || c.Message.Content != nil ||
		len(c.Message.ToolCalls) != 1 || c.Message.ToolCalls[0].Function.Name != "lookup" ||
		out.Usage.TotalTokens != 24 || out.Usage.PromptTokensDetails.CachedTokens != 16 {
		t.Fatalf("response = %s", body)
	}

	got := p.Requests()[0]
	if got.Model != "m1" || got.MaxTokens != 64 || *got.Temperature != 0.3 {
		t.Fatalf("engine request = %+v", got)
	}
	if got.Messages[1].Content != "hello" {
		t.Fatalf("content parts not flattened: %q", got.Messages[1].Content)
	}
	if string(got.Extra["tool_choice"]) != `"required"` || string(got.Extra["seed"]) != `5` {
		t.Fatalf("extra not passed through: %v", got.Extra)
	}
	if _, leaked := got.Extra["tools"]; leaked {
		t.Fatal("modelled fields must not appear in Extra")
	}
}

func TestStreamingIsSynthesisedFromFullResponse(t *testing.T) {
	srv := newServer(t, mock.New(mock.Text("hi there")))
	res, body := post(t, srv.URL+"/chat/completions", // also checks the /v1-less route
		`{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"x"}]}`)
	if res.StatusCode != 200 || res.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d, type %q", res.StatusCode, res.Header.Get("Content-Type"))
	}

	var payloads []string
	sc := bufio.NewScanner(bytes.NewReader(body))
	for sc.Scan() {
		if line := sc.Text(); strings.HasPrefix(line, "data: ") {
			payloads = append(payloads, strings.TrimPrefix(line, "data: "))
		}
	}
	if len(payloads) != 4 || payloads[3] != "[DONE]" {
		t.Fatalf("payloads = %q", payloads)
	}
	var first, last, usage struct {
		Object  string `json:"object"`
		Choices []struct {
			Delta        map[string]any `json:"delta"`
			FinishReason *string        `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	for i, dst := range []any{&first, &last, &usage} {
		if err := json.Unmarshal([]byte(payloads[i]), dst); err != nil {
			t.Fatalf("chunk %d: %v", i, err)
		}
	}
	if first.Object != "chat.completion.chunk" || first.Choices[0].Delta["content"] != "hi there" || first.Choices[0].Delta["role"] != "assistant" {
		t.Fatalf("first = %s", payloads[0])
	}
	if last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "stop" {
		t.Fatalf("last = %s", payloads[1])
	}
	if len(usage.Choices) != 0 || usage.Usage == nil || usage.Usage.TotalTokens != 15 {
		t.Fatalf("usage chunk = %s", payloads[2])
	}
}

func TestAuth(t *testing.T) {
	srv := newServer(t, mock.New(mock.Text("a"), mock.Text("b")), "s3cret", "other")
	const body = `{"model":"m","messages":[{"role":"user","content":"x"}]}`

	if res, _ := post(t, srv.URL+"/v1/chat/completions", body); res.StatusCode != 401 {
		t.Fatalf("no token: status %d", res.StatusCode)
	}
	if res, _ := post(t, srv.URL+"/v1/chat/completions", body, "Authorization", "Bearer nope"); res.StatusCode != 401 {
		t.Fatalf("wrong token: status %d", res.StatusCode)
	}
	if res, b := post(t, srv.URL+"/v1/chat/completions", body, "Authorization", "Bearer s3cret"); res.StatusCode != 200 {
		t.Fatalf("right token: status %d: %s", res.StatusCode, b)
	}
	if res, b := post(t, srv.URL+"/v1/chat/completions", body, "Authorization", "Bearer other"); res.StatusCode != 200 {
		t.Fatalf("second token: status %d: %s", res.StatusCode, b)
	}
	// Health stays open so load balancers can probe it.
	res, err := http.Get(srv.URL + "/healthz")
	if err != nil || res.StatusCode != 200 {
		t.Fatalf("healthz: %v %v", err, res)
	}
	res.Body.Close()
}

func TestBadRequests(t *testing.T) {
	srv := newServer(t, mock.New())
	for name, body := range map[string]string{
		"not json":        `{`,
		"no messages":     `{"model":"m"}`,
		"empty messages":  `{"model":"m","messages":[]}`,
		"non-function":    `{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"web_search"}]}`,
		"image part":      `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url"}]}]}`,
		"bad temperature": `{"model":"m","temperature":"hot","messages":[{"role":"user","content":"x"}]}`,
	} {
		res, b := post(t, srv.URL+"/v1/chat/completions", body)
		if res.StatusCode != 400 || !strings.Contains(string(b), `"error"`) {
			t.Errorf("%s: status %d body %s", name, res.StatusCode, b)
		}
	}

	res, err := http.Get(srv.URL + "/v1/chat/completions")
	if err != nil || res.StatusCode != 405 {
		t.Fatalf("GET: %v %v", err, res)
	}
	res.Body.Close()
}

type statusErr struct {
	code  int
	retry time.Duration
}

func (e statusErr) Error() string                     { return "boom" }
func (e statusErr) StatusCode() int                   { return e.code }
func (e statusErr) RetryAfterDuration() time.Duration { return e.retry }

func TestUpstreamErrorMapping(t *testing.T) {
	const body = `{"model":"m","messages":[{"role":"user","content":"x"}]}`
	cases := []struct {
		name      string
		err       error
		want      int
		retryHdr  string
		notInBody string
	}{
		{"rate limit passes through with Retry-After", statusErr{429, 1500 * time.Millisecond}, 429, "2", ""},
		{"bad request passes through", statusErr{400, 0}, 400, "", ""},
		{"upstream 401 is our fault, not the client's", statusErr{401, 0}, 502, "", ""},
		{"upstream 500 becomes 502", statusErr{500, 0}, 502, "", ""},
		{"plain error is generic", errors.New("dial tcp 10.0.0.5: secret-host refused"), 502, "", "secret-host"},
	}
	for _, c := range cases {
		srv := newServer(t, mock.New(mock.Reply{Err: c.err}))
		res, b := post(t, srv.URL+"/v1/chat/completions", body)
		if res.StatusCode != c.want || res.Header.Get("Retry-After") != c.retryHdr {
			t.Errorf("%s: status %d retry %q, body %s", c.name, res.StatusCode, res.Header.Get("Retry-After"), b)
		}
		if c.notInBody != "" && strings.Contains(string(b), c.notInBody) {
			t.Errorf("%s: internal detail leaked: %s", c.name, b)
		}
	}
}

func TestGuardRejectionReachesTheClientWithAUsefulMessage(t *testing.T) {
	bad := mock.Reply{Response: &core.Response{
		Message:      core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "1", Name: "rm_rf", Arguments: `{}`}}},
		FinishReason: "tool_calls",
	}}
	engine := &core.Engine{
		Provider:    mock.New(bad, bad),
		Layers:      []core.Layer{guard.New(guard.Config{})},
		MaxAttempts: 2,
	}
	srv := httptest.NewServer((&proxy.Server{Engine: engine}).Handler())
	defer srv.Close()

	res, b := post(t, srv.URL+"/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"safe","parameters":{"type":"object"}}}]}`)
	if res.StatusCode != 502 || !strings.Contains(string(b), "unknown tool") || !strings.Contains(string(b), "only the tools offered") {
		t.Fatalf("status %d body %s", res.StatusCode, b)
	}
}
