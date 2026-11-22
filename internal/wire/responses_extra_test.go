package wire

import (
	"encoding/json"
	"testing"
)

func TestResponsesKeepsReasoningEffortAndUser(t *testing.T) {
	body := `{"model":"gpt-5-nano","input":"hi","store":false,"user":"alice","reasoning":{"effort":"minimal","summary":"auto"}}`
	req, err := DecodeResponsesRequest([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var effort, user string
	json.Unmarshal(req.Extra["reasoning_effort"], &effort)
	json.Unmarshal(req.Extra["user"], &user)
	if effort != "minimal" || user != "alice" {
		t.Fatalf("effort %q user %q extra %v", effort, user, req.Extra)
	}
}

func TestResponsesWithoutThemHasNoExtraForThem(t *testing.T) {
	req, err := DecodeResponsesRequest([]byte(`{"model":"m","input":"hi","reasoning":{"summary":"auto"},"user":null}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := req.Extra["reasoning_effort"]; ok {
		t.Fatal("effort invented")
	}
	if _, ok := req.Extra["user"]; ok {
		t.Fatal("null user forwarded")
	}
}
