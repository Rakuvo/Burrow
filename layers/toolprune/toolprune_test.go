package toolprune

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/rakuvo/burrow/core"
)

func tool(name, desc string) core.ToolSpec {
	return core.ToolSpec{Name: name, Description: desc, Schema: `{"type":"object"}`}
}

func catalog() []core.ToolSpec {
	ts := []core.ToolSpec{
		tool("send_tweet", "Post a tweet to the timeline"),
		tool("list_files", "List the files in the current directory"),
		tool("book_flight", "Book a flight ticket between two cities"),
		tool("get_stock_price", "Current price of a stock symbol"),
		tool("cd", "Change the current directory"),
	}
	for i := 0; i < 20; i++ {
		ts = append(ts, tool(fmt.Sprintf("filler_op_%d", i), fmt.Sprintf("Unrelated operation number %d", i)))
	}
	return ts
}

func run(t *testing.T, cfg Config, req core.Request) (*core.Step, []string) {
	t.Helper()
	s := &core.Step{Original: req, Request: req.Clone()}
	if err := New(cfg).Before(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, tl := range s.Request.Tools {
		names = append(names, tl.Name)
	}
	return s, names
}

func has(names []string, n string) bool {
	for _, x := range names {
		if x == n {
			return true
		}
	}
	return false
}

func user(c string) core.Message { return core.Message{Role: core.RoleUser, Content: c} }

func TestKeepsRelevantAndDropsRest(t *testing.T) {
	_, names := run(t, Config{Keep: 3}, core.Request{Tools: catalog(), Messages: []core.Message{user("please book a flight to Paris and post a tweet about it")}})
	if len(names) != 3 || !has(names, "book_flight") || !has(names, "send_tweet") {
		t.Fatalf("got %v", names)
	}
}

func TestOriginalOrderKept(t *testing.T) {
	_, names := run(t, Config{Keep: 2}, core.Request{Tools: catalog(), Messages: []core.Message{user("tweet about the flight")}})
	if len(names) != 2 || names[0] != "send_tweet" || names[1] != "book_flight" {
		t.Fatalf("got %v", names)
	}
}

func TestUsedToolsAlwaysKept(t *testing.T) {
	msgs := []core.Message{
		user("list files"),
		{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "1", Name: "get_stock_price", Arguments: "{}"}}},
		{Role: core.RoleTool, ToolCallID: "1", Content: "ok"},
		user("now tweet it"),
	}
	_, names := run(t, Config{Keep: 1}, core.Request{Tools: catalog(), Messages: msgs})
	if !has(names, "get_stock_price") || !has(names, "send_tweet") {
		t.Fatalf("got %v", names)
	}
}

func TestForcedToolChoiceKept(t *testing.T) {
	extra := map[string]json.RawMessage{"tool_choice": json.RawMessage(`{"type":"function","function":{"name":"filler_op_7"}}`)}
	_, names := run(t, Config{Keep: 1}, core.Request{Tools: catalog(), Extra: extra, Messages: []core.Message{user("tweet it")}})
	if !has(names, "filler_op_7") {
		t.Fatalf("got %v", names)
	}
}

func TestShortListUntouched(t *testing.T) {
	s, names := run(t, Config{Keep: 2}, core.Request{Tools: catalog()[:4], Messages: []core.Message{user("tweet")}})
	if len(names) != 4 || len(s.Request.Tools) != 4 {
		t.Fatalf("got %v", names)
	}
}

func TestNoMatchLeavesAll(t *testing.T) {
	_, names := run(t, Config{Keep: 2}, core.Request{Tools: catalog(), Messages: []core.Message{user("zzzz qqqq")}})
	if len(names) != len(catalog()) {
		t.Fatalf("got %d tools", len(names))
	}
}

func TestOffByDefaultAndNoUserMessage(t *testing.T) {
	_, names := run(t, Config{}, core.Request{Tools: catalog(), Messages: []core.Message{user("tweet")}})
	if len(names) != len(catalog()) {
		t.Fatal("zero config changed the request")
	}
	_, names = run(t, Config{Keep: 2}, core.Request{Tools: catalog()})
	if len(names) != len(catalog()) {
		t.Fatal("no user message changed the request")
	}
}

func TestDoesNotMutateOriginalAndNotes(t *testing.T) {
	s, names := run(t, Config{Keep: 2}, core.Request{Tools: catalog(), Messages: []core.Message{user("book a flight")}})
	if len(s.Original.Tools) != len(catalog()) || len(names) >= len(catalog()) {
		t.Fatal("original changed or nothing pruned")
	}
}

func TestTokensSplitsCamelAndSnake(t *testing.T) {
	got := tokens("getStockPrice list_files")
	for _, w := range []string{"stock", "price", "list", "file", "files"} {
		if !got[w] {
			t.Errorf("missing %q in %v", w, got)
		}
	}
}
