package wire

import (
	"encoding/json"
	"testing"

	"github.com/rakuvo/burrow/core"
)

// These decoders parse bytes straight off the network. They must never panic,
// and whatever they accept must be safe to encode again.

func FuzzDecodeRequest(f *testing.F) {
	for _, s := range []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"x"}]}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto","stream":true}`,
		`{"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`,
		`[]`, `null`, `{"messages":"x"}`,
	} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, in []byte) {
		p, err := DecodeRequest(in)
		if err != nil {
			return
		}
		_, _ = EncodeRequest(p.Request, RequestOptions{})
	})
}

func FuzzDecodeResponsesRequest(f *testing.F) {
	for _, s := range []string{
		`{"model":"m","input":"hi"}`,
		`{"model":"m","input":[{"role":"developer","content":"x"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"ok"},{"type":"reasoning","summary":[]}],"tools":[{"type":"function","name":"f","parameters":{}}],"tool_choice":{"type":"function","name":"f"}}`,
		`{"input":[{"role":"user","content":[{"type":"input_text","text":"a"},{"type":"refusal","refusal":"b"}]}],"store":false,"top_p":1}`,
		`{"input":[null]}`, `{"input":[[]]}`, `{"tools":"x"}`, `{"input":"x","tool_choice":7}`,
	} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, in []byte) {
		req, err := DecodeResponsesRequest(in)
		if err != nil {
			return
		}
		if len(req.Messages) == 0 {
			t.Fatalf("accepted a request with no messages: %q", in)
		}
		_, _ = EncodeRequest(*req, RequestOptions{})
	})
}

func FuzzDecodeResponse(f *testing.F) {
	for _, s := range []string{
		`{"model":"m","choices":[{"message":{"content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1}}`,
		`{"choices":[{"message":{"content":null,"tool_calls":[{"id":"1","function":{"name":"f","arguments":"{}"}}]}}]}`,
		`{"choices":[]}`, `{}`, `[]`,
	} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, in []byte) {
		r, err := DecodeResponse(in)
		if err != nil {
			return
		}
		// Whatever came in must be re-encodable for our own clients.
		if _, err := EncodeResponse(r, "id", 0); err != nil {
			t.Fatalf("accepted a response it cannot encode: %v", err)
		}
		if _, err := EncodeResponses(r, "id", 0); err != nil {
			t.Fatalf("accepted a response it cannot encode (responses): %v", err)
		}
		if _, err := StreamChunks(r, "id", 0, true); err != nil {
			t.Fatalf("accepted a response it cannot stream: %v", err)
		}
	})
}

// A request that decodes through the chat path and the Responses path must
// mean the same thing to the engine.
func TestChatAndResponsesDecodeToTheSameRequest(t *testing.T) {
	chat, err := DecodeRequest([]byte(`{"model":"m","temperature":0.5,"max_tokens":9,
		"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}],
		"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"c1","content":"r"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := DecodeResponsesRequest([]byte(`{"model":"m","temperature":0.5,"max_output_tokens":9,
		"tools":[{"type":"function","name":"f","description":"d","parameters":{"type":"object"}}],
		"input":[{"role":"system","content":"s"},{"role":"user","content":"u"},
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
		{"type":"function_call_output","call_id":"c1","output":"r"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(chat.Request)
	b, _ := json.Marshal(*resp)
	if string(a) != string(b) {
		t.Fatalf("the two front doors disagree:\nchat:      %s\nresponses: %s", a, b)
	}
	var _ core.Request = chat.Request
}
