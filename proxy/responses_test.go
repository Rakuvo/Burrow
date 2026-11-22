package proxy_test

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/layers/guard"
	"github.com/rakuvo/burrow/providers/mock"
	"github.com/rakuvo/burrow/proxy"
)

const weatherFlat = `{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object","required":["city","days"],"properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}`

func callReply(args string) mock.Reply {
	return mock.Reply{Response: &core.Response{
		Model:        "gpt-x",
		Message:      core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "call_1", Name: "get_weather", Arguments: args}}},
		FinishReason: "tool_calls",
		Usage:        core.Usage{InputTokens: 50, OutputTokens: 8, CachedTokens: 10},
	}}
}

func TestResponsesFirstTurnLikeBFCL(t *testing.T) {
	p := mock.New(callReply(`{"city":"Pune","days":3}`))
	srv := newServer(t, p)
	res, body := post(t, srv.URL+"/v1/responses", `{
		"model":"gpt-x","store":false,"temperature":0.001,"max_output_tokens":77,"top_p":0.9,"parallel_tool_calls":true,
		"include":["reasoning.encrypted_content"],"reasoning":{"summary":"auto"},
		"tools":[`+weatherFlat+`],
		"input":[{"role":"developer","content":"be terse"},{"role":"user","content":"weather in Pune for 3 days"}]}`)
	if res.StatusCode != 200 {
		t.Fatalf("status %d: %s", res.StatusCode, body)
	}

	var out struct {
		ID     string `json:"id"`
		Object string `json:"object"`
		Status string `json:"status"`
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			Details      struct {
				Cached int `json:"cached_tokens"`
			} `json:"input_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "response" || out.Status != "completed" || !strings.HasPrefix(out.ID, "resp_") ||
		len(out.Output) != 1 || out.Output[0].Type != "function_call" || out.Output[0].Name != "get_weather" ||
		out.Output[0].CallID != "call_1" || out.Output[0].Arguments != `{"city":"Pune","days":3}` {
		t.Fatalf("response = %s", body)
	}
	if out.Usage.InputTokens != 50 || out.Usage.OutputTokens != 8 || out.Usage.Details.Cached != 10 {
		t.Fatalf("usage = %+v", out.Usage)
	}

	got := p.Requests()[0]
	if got.Model != "gpt-x" || got.MaxTokens != 77 || *got.Temperature != 0.001 {
		t.Fatalf("engine request = %+v", got)
	}
	if got.Messages[0].Role != core.RoleSystem || got.Messages[0].Content != "be terse" {
		t.Fatalf("developer role must reach the upstream as system: %+v", got.Messages[0])
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "get_weather" || !strings.Contains(got.Tools[0].Schema, "required") {
		t.Fatalf("tools = %+v", got.Tools)
	}
	if string(got.Extra["top_p"]) != "0.9" || string(got.Extra["parallel_tool_calls"]) != "true" {
		t.Fatalf("extra = %v", got.Extra)
	}
	for _, leaked := range []string{"store", "include", "reasoning"} {
		if _, ok := got.Extra[leaked]; ok {
			t.Errorf("%s is Responses-only and must not be forwarded to a chat upstream", leaked)
		}
	}
}

func TestResponsesMultiTurnEchoedItems(t *testing.T) {
	p := mock.New(mock.Text("It is 31C in Pune."))
	srv := newServer(t, p)
	// What BFCL sends on turn 2: its own earlier output echoed back, then the tool result.
	res, body := post(t, srv.URL+"/v1/responses", `{"model":"m","tools":[`+weatherFlat+`],"input":[
		{"role":"user","content":"weather in Pune?"},
		{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Checking.","annotations":[]}]},
		{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_9","name":"get_weather","arguments":"{\"city\":\"Pune\",\"days\":1}"},
		{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"abc"},
		{"type":"function_call_output","call_id":"call_9","output":"31C"}]}`)
	if res.StatusCode != 200 {
		t.Fatalf("status %d: %s", res.StatusCode, body)
	}
	m := p.Requests()[0].Messages
	if len(m) != 3 {
		t.Fatalf("messages = %+v", m)
	}
	if m[1].Role != core.RoleAssistant || m[1].Content != "Checking." || len(m[1].ToolCalls) != 1 || m[1].ToolCalls[0].ID != "call_9" {
		t.Fatalf("the call must join the assistant turn that preceded it: %+v", m[1])
	}
	if m[2].Role != core.RoleTool || m[2].ToolCallID != "call_9" || m[2].Content != "31C" {
		t.Fatalf("tool result = %+v", m[2])
	}
	if !strings.Contains(string(body), `"output_text"`) || !strings.Contains(string(body), "It is 31C in Pune.") {
		t.Fatalf("text answer should come back as an output_text message: %s", body)
	}
}

func TestResponsesStringInputInstructionsAndToolChoice(t *testing.T) {
	p := mock.New(mock.Text("ok"))
	srv := newServer(t, p)
	res, body := post(t, srv.URL+"/responses", `{"model":"m","instructions":"sys","input":"hello",
		"tools":[`+weatherFlat+`],"tool_choice":{"type":"function","name":"get_weather"}}`)
	if res.StatusCode != 200 {
		t.Fatalf("status %d: %s", res.StatusCode, body)
	}
	got := p.Requests()[0]
	if len(got.Messages) != 2 || got.Messages[0].Role != core.RoleSystem || got.Messages[1].Content != "hello" {
		t.Fatalf("messages = %+v", got.Messages)
	}
	if string(got.Extra["tool_choice"]) != `{"function":{"name":"get_weather"},"type":"function"}` {
		t.Fatalf("tool_choice must be converted to the chat form: %s", got.Extra["tool_choice"])
	}
}

func TestResponsesRefusesWhatItCannotDoFaithfully(t *testing.T) {
	srv := newServer(t, mock.New())
	const in = `"input":"hi"`
	for name, body := range map[string]string{
		"streaming":       `{"model":"m",` + in + `,"stream":true}`,
		"stateful":        `{"model":"m",` + in + `,"previous_response_id":"resp_1"}`,
		"background":      `{"model":"m",` + in + `,"background":true}`,
		"structured text": `{"model":"m",` + in + `,"text":{"format":{"type":"json_object"}}}`,
		"hosted tool":     `{"model":"m",` + in + `,"tools":[{"type":"web_search"}]}`,
		"odd item":        `{"model":"m","input":[{"type":"computer_call"}]}`,
		"image part":      `{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]}`,
		"odd tool_choice": `{"model":"m",` + in + `,"tool_choice":{"type":"allowed_tools"}}`,
		"missing input":   `{"model":"m"}`,
		"not json":        `{`,
	} {
		res, b := post(t, srv.URL+"/v1/responses", body)
		if res.StatusCode != 400 || !strings.Contains(string(b), `"error"`) {
			t.Errorf("%s: status %d body %s", name, res.StatusCode, b)
		}
	}
}

func TestResponsesTruncationIsMarkedIncomplete(t *testing.T) {
	cut := mock.Text("partial")
	cut.Response.FinishReason = "length"
	srv := newServer(t, mock.New(cut))
	_, body := post(t, srv.URL+"/v1/responses", `{"model":"m","input":"hi"}`)
	if !strings.Contains(string(body), `"status":"incomplete"`) || !strings.Contains(string(body), "max_output_tokens") {
		t.Fatalf("body = %s", body)
	}
}

func TestResponsesRunsThroughTheGuard(t *testing.T) {
	engine := &core.Engine{Provider: mock.New(callReply(`{"city":"Pune","days":"3"}`)), Layers: []core.Layer{guard.New(guard.Config{})}}
	srv := httptest.NewServer((&proxy.Server{Engine: engine}).Handler())
	defer srv.Close()
	res, body := post(t, srv.URL+"/v1/responses", `{"model":"m","tools":[`+weatherFlat+`],"input":"weather in Pune for 3 days"}`)
	if res.StatusCode != 200 || !strings.Contains(string(body), `\"days\":3`) {
		t.Fatalf("the guard should repair \"3\" to 3 on this endpoint too: %d %s", res.StatusCode, body)
	}
}

func TestResponsesNeedsAuthWhenConfigured(t *testing.T) {
	srv := newServer(t, mock.New(mock.Text("a")), "s3cret")
	if res, _ := post(t, srv.URL+"/v1/responses", `{"model":"m","input":"hi"}`); res.StatusCode != 401 {
		t.Fatalf("status %d", res.StatusCode)
	}
	if res, b := post(t, srv.URL+"/v1/responses", `{"model":"m","input":"hi"}`, "Authorization", "Bearer s3cret"); res.StatusCode != 200 {
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}
}
