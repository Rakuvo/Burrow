package loopguard

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/rakuvo/burrow/core"
)

var tools = []core.ToolSpec{{Name: "search", Schema: "{}"}, {Name: "open", Schema: "{}"}}

func user(c string) core.Message { return core.Message{Role: core.RoleUser, Content: c} }

func call(id, name, args string) core.Message {
	return core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: id, Name: name, Arguments: args}}}
}

func res(id, c string) core.Message {
	return core.Message{Role: core.RoleTool, ToolCallID: id, Content: c}
}

// turns builds a user message followed by n rounds of (call, result) from the generator.
func turns(n int, gen func(i int) (name, args, result string)) []core.Message {
	msgs := []core.Message{user("find it")}
	for i := 0; i < n; i++ {
		name, args, r := gen(i)
		id := fmt.Sprintf("c%d", i)
		msgs = append(msgs, call(id, name, args), res(id, r))
	}
	return msgs
}

func same(name, args, r string) func(int) (string, string, string) {
	return func(int) (string, string, string) { return name, args, r }
}

// scripted is a provider that answers with a fixed message and records what it was sent.
type scripted struct {
	reply core.Message
	got   []core.Request
}

func (*scripted) Name() string { return "scripted" }
func (p *scripted) Complete(_ context.Context, r core.Request) (*core.Response, error) {
	p.got = append(p.got, r)
	return &core.Response{Message: p.reply, FinishReason: "tool_calls", Usage: core.Usage{InputTokens: 10, OutputTokens: 5}}, nil
}

func replyCall(name, args string) core.Message {
	return core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "new", Name: name, Arguments: args}}}
}

func drive(t *testing.T, cfg Config, msgs []core.Message, reply core.Message, extra map[string]json.RawMessage) (*core.Response, *scripted, []string) {
	t.Helper()
	p := &scripted{reply: reply}
	tr := &core.MemoryTracer{}
	e := &core.Engine{Provider: p, Layers: []core.Layer{New(cfg)}, Tracer: tr}
	resp, err := e.Complete(context.Background(), core.Request{Model: "m", Messages: msgs, Tools: tools, Extra: extra})
	if err != nil {
		t.Fatal(err)
	}
	var notes []string
	for _, ev := range tr.Events() {
		for _, n := range ev.Notes {
			notes = append(notes, n.Message)
		}
	}
	return resp, p, notes
}

func TestZeroConfigDoesNothing(t *testing.T) {
	msgs := turns(12, same("search", `{"q":"a"}`, "nothing"))
	resp, p, notes := drive(t, Config{}, msgs, replyCall("search", `{"q":"a"}`), nil)
	if len(resp.Message.ToolCalls) != 1 || len(p.got) != 1 || len(notes) != 0 {
		t.Fatalf("zero config must not intervene: %+v %d %v", resp.Message, len(p.got), notes)
	}
}

func TestNoProgressHistoryGetsANudge(t *testing.T) {
	msgs := turns(3, same("search", `{"q":"a"}`, "no results"))
	_, p, notes := drive(t, Defaults(), msgs, replyCall("open", `{"id":1}`), nil)
	last := p.got[0].Messages[len(p.got[0].Messages)-1]
	if last.Role != core.RoleSystem || !strings.Contains(last.Content, "search was called 3 times") {
		t.Fatalf("expected a nudge as the last message, got %+v", last)
	}
	if len(notes) != 1 || !strings.Contains(notes[0], "stuck") {
		t.Fatalf("notes: %v", notes)
	}
}

func TestNudgeNotAddedForHealthyConversation(t *testing.T) {
	msgs := turns(6, func(i int) (string, string, string) {
		return "search", fmt.Sprintf(`{"q":"term %d"}`, i), fmt.Sprintf("result %d", i)
	})
	_, p, notes := drive(t, Defaults(), msgs, replyCall("search", `{"q":"term 6"}`), nil)
	if len(p.got[0].Messages) != len(msgs) || len(notes) != 0 {
		t.Fatalf("healthy runs must be untouched: %v", notes)
	}
}

func TestStopLevelReplacesTheLoopingCall(t *testing.T) {
	msgs := turns(5, same("search", `{"q":"a"}`, "no results"))
	resp, p, notes := drive(t, Defaults(), msgs, replyCall("search", `{"q":"a"}`), nil)
	if len(resp.Message.ToolCalls) != 0 || resp.FinishReason != "stop" {
		t.Fatalf("the loop should have been stopped: %+v", resp)
	}
	if !strings.Contains(resp.Message.Content, "Stopped by Rakuvo") || !strings.Contains(resp.Message.Content, "search was called 5 times") {
		t.Fatalf("halt message: %q", resp.Message.Content)
	}
	if len(p.got) != 1 {
		t.Fatalf("a halt must not trigger a retry, provider called %d times", len(p.got))
	}
	if resp.Usage.InputTokens != 10 {
		t.Fatal("usage of the replaced call must still be reported")
	}
	if !strings.Contains(strings.Join(notes, "|"), "stopped the loop") {
		t.Fatalf("notes: %v", notes)
	}
}

func TestWarnOnlyNeverAltersTheResponse(t *testing.T) {
	cfg := Defaults()
	cfg.Action = WarnOnly
	msgs := turns(6, same("search", `{"q":"a"}`, "no results"))
	resp, _, _ := drive(t, cfg, msgs, replyCall("search", `{"q":"a"}`), nil)
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatal("WarnOnly must leave the response alone")
	}
}

func TestPredictiveRetryWithNudge(t *testing.T) {
	// two identical calls in history: the model's new identical call would be the third
	msgs := turns(2, same("search", `{"q":"a"}`, "no results"))
	cfg := Defaults()
	cfg.NoProgressWarn, cfg.NoProgressStop = 0, 0 // isolate the repeat detector
	cfg.RepeatWarn, cfg.RepeatStop = 3, 0
	resp, p, notes := drive(t, cfg, msgs, replyCall("search", `{"q":"a"}`), nil)
	if len(p.got) != 2 {
		t.Fatalf("expected one retry with a nudge, provider called %d times", len(p.got))
	}
	first, second := p.got[0].Messages, p.got[1].Messages
	if len(second) != len(first)+1 || second[len(second)-1].Role != core.RoleSystem {
		t.Fatal("second attempt should carry the nudge")
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatal("below the stop level the call is passed through after the nudge")
	}
	if resp.Usage.InputTokens != 20 {
		t.Fatalf("usage should cover both attempts, got %d", resp.Usage.InputTokens)
	}
	if len(notes) < 2 {
		t.Fatalf("notes: %v", notes)
	}
}

func TestDifferentResultsAreNotNoProgress(t *testing.T) {
	// polling: same call, results change. Only the (higher) repeat limit applies.
	msgs := turns(3, func(i int) (string, string, string) {
		return "search", `{"q":"job"}`, fmt.Sprintf("running %d%%", i*25)
	})
	resp, _, notes := drive(t, Defaults(), msgs, replyCall("search", `{"q":"job"}`), nil)
	if len(resp.Message.ToolCalls) != 1 || len(notes) != 0 {
		t.Fatalf("polling with changing results must not trip: %v", notes)
	}
}

func TestAllowRepeatExemptsPolling(t *testing.T) {
	cfg := Defaults()
	cfg.AllowRepeat = []string{"search"}
	msgs := turns(11, func(i int) (string, string, string) { return "search", `{"q":"job"}`, fmt.Sprintf("ok %d", i) })
	resp, _, _ := drive(t, cfg, msgs, replyCall("search", `{"q":"job"}`), nil)
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatal("allowed tool was stopped")
	}
	cfg.AllowRepeat = nil
	resp, _, _ = drive(t, cfg, msgs, replyCall("search", `{"q":"job"}`), nil)
	if len(resp.Message.ToolCalls) != 0 {
		t.Fatal("without the exemption 12 identical calls should be stopped")
	}
}

func TestArgumentOrderAndSpacingDoNotHideALoop(t *testing.T) {
	l := New(Defaults())
	a := l.fingerprint(core.ToolCall{Name: "f", Arguments: `{"a":1,"b":[1,2]}`})
	b := l.fingerprint(core.ToolCall{Name: "f", Arguments: "{ \"b\": [1, 2],\n \"a\": 1 }"})
	if a != b {
		t.Fatal("same arguments in a different order and spacing must match")
	}
	if a == l.fingerprint(core.ToolCall{Name: "g", Arguments: `{"a":1,"b":[1,2]}`}) {
		t.Fatal("different tools must not match")
	}
	if a == l.fingerprint(core.ToolCall{Name: "f", Arguments: `{"a":2,"b":[1,2]}`}) {
		t.Fatal("different values must not match")
	}
}

func TestIgnoreArgsAndKeyArgs(t *testing.T) {
	cfg := Defaults()
	cfg.IgnoreArgs = []string{"ts"}
	cfg.KeyArgs = map[string][]string{"open": {"id"}}
	l := New(cfg)
	if l.fingerprint(core.ToolCall{Name: "search", Arguments: `{"q":"x","ts":1}`}) != l.fingerprint(core.ToolCall{Name: "search", Arguments: `{"q":"x","ts":2}`}) {
		t.Fatal("ignored argument still changes the fingerprint")
	}
	if l.fingerprint(core.ToolCall{Name: "open", Arguments: `{"id":7,"trace":"a"}`}) != l.fingerprint(core.ToolCall{Name: "open", Arguments: `{"id":7,"trace":"b"}`}) {
		t.Fatal("only key arguments should count")
	}
	if l.fingerprint(core.ToolCall{Name: "open", Arguments: `{"id":7}`}) == l.fingerprint(core.ToolCall{Name: "open", Arguments: `{"id":8}`}) {
		t.Fatal("a key argument change must change the fingerprint")
	}
}

func TestCycleDetection(t *testing.T) {
	msgs := turns(6, func(i int) (string, string, string) {
		if i%2 == 0 {
			return "search", `{"q":"a"}`, fmt.Sprintf("r%d", i)
		}
		return "open", `{"id":1}`, fmt.Sprintf("r%d", i)
	})
	// results change every time, so no-progress stays quiet; the ping-pong is the signal
	resp, _, notes := drive(t, Defaults(), msgs, replyCall("search", `{"q":"a"}`), nil)
	if !strings.Contains(strings.Join(notes, "|"), "cycle search -> open") {
		t.Fatalf("expected a cycle finding, notes: %v", notes)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("a cycle of 3-4 reps is only a warning: %+v", resp.Message)
	}
}

func TestSingleRepeatedCallIsNotACycle(t *testing.T) {
	l := New(Defaults())
	hist := l.history(turns(8, same("search", `{"q":"a"}`, "x")))
	if got := l.cycles(hist); len(got) != 0 {
		t.Fatalf("one call repeated is not a cycle: %+v", got)
	}
}

func TestTextRepetition(t *testing.T) {
	sentence := "I could not find the document you asked about in the search results, so let me try searching once more."
	msgs := []core.Message{user("find it")}
	for i := 0; i < 3; i++ {
		msgs = append(msgs, core.Message{Role: core.RoleAssistant, Content: sentence}, user("and?"))
	}
	f := New(Defaults()).analyze(msgs, nil)
	if len(f) != 1 || f[0].Kind != "text" || f[0].Count != 3 {
		t.Fatalf("findings: %+v", f)
	}
	msgs[3].Content = "Completely different content that talks about something else entirely, with new words."
	if f := New(Defaults()).analyze(msgs, nil); len(f) != 0 {
		t.Fatalf("different messages are not a repeat: %+v", f)
	}
}

func TestSimilarity(t *testing.T) {
	if similarity("hello world, hello", "hello world, hello") != 1 {
		t.Fatal("identical strings")
	}
	if similarity("abcdef", "uvwxyz") != 0 {
		t.Fatal("disjoint strings")
	}
	if s := similarity("the quick brown fox", "the quick brown foxes"); s < 0.7 || s >= 1 {
		t.Fatalf("near strings: %v", s)
	}
}

func TestMaxToolCallsCap(t *testing.T) {
	cfg := Config{MaxToolCalls: 5}
	msgs := turns(5, func(i int) (string, string, string) { return "search", fmt.Sprintf(`{"q":"%d"}`, i), "r" })
	resp, _, notes := drive(t, cfg, msgs, replyCall("search", `{"q":"new"}`), nil)
	if len(resp.Message.ToolCalls) != 0 || !strings.Contains(resp.Message.Content, "over the limit of 5") {
		t.Fatalf("cap not enforced: %+v", resp.Message)
	}
	if !strings.Contains(strings.Join(notes, "|"), "stopped the loop") {
		t.Fatalf("notes: %v", notes)
	}
	resp, _, _ = drive(t, cfg, msgs[:7], replyCall("search", `{"q":"new"}`), nil) // 3 calls + 1 new = 4
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatal("under the cap the call must pass")
	}
}

func TestNeverHaltsPlainTextOrForcedTools(t *testing.T) {
	msgs := turns(6, same("search", `{"q":"a"}`, "no results"))
	text := core.Message{Role: core.RoleAssistant, Content: "I give up, here is what I know."}
	resp, _, _ := drive(t, Defaults(), msgs, text, nil)
	if resp.Message.Content != text.Content {
		t.Fatal("a plain-text answer must pass untouched")
	}
	forced := map[string]json.RawMessage{"tool_choice": json.RawMessage(`"required"`)}
	resp, p, notes := drive(t, Defaults(), msgs, replyCall("search", `{"q":"a"}`), forced)
	if len(resp.Message.ToolCalls) != 1 || len(p.got) != 1 {
		t.Fatalf("a forced tool_choice must not be overridden or retried (calls %d)", len(p.got))
	}
	if !strings.Contains(strings.Join(notes, "|"), "caller dictated") {
		t.Fatalf("notes: %v", notes)
	}
}

func TestNoToolsMeansNoAction(t *testing.T) {
	msgs := turns(6, same("search", `{"q":"a"}`, "x"))
	p := &scripted{reply: replyCall("search", `{"q":"a"}`)}
	e := &core.Engine{Provider: p, Layers: []core.Layer{New(Defaults())}}
	resp, err := e.Complete(context.Background(), core.Request{Model: "m", Messages: msgs})
	if err != nil || len(resp.Message.ToolCalls) != 1 || len(p.got) != 1 {
		t.Fatal("without tools the layer must stay out of the way")
	}
}

func TestIdenticalCallsInOneMessageCountOnce(t *testing.T) {
	msgs := []core.Message{user("go"), {Role: core.RoleAssistant, ToolCalls: []core.ToolCall{
		{ID: "a", Name: "search", Arguments: `{"q":"x"}`}, {ID: "b", Name: "search", Arguments: `{"q":"x"}`},
		{ID: "c", Name: "search", Arguments: `{"q":"x"}`}}},
		res("a", "r"), res("b", "r"), res("c", "r")}
	if f := New(Defaults()).analyze(msgs, nil); len(f) != 0 {
		t.Fatalf("three identical calls in one message are one step, not a loop: %+v", f)
	}
}

func TestDoesNotModifyCallersMessages(t *testing.T) {
	msgs := turns(5, same("search", `{"q":"a"}`, "x"))
	before := fmt.Sprint(msgs)
	drive(t, Defaults(), msgs, replyCall("search", `{"q":"a"}`), nil)
	if fmt.Sprint(msgs) != before {
		t.Fatal("caller's messages were modified")
	}
}

func TestWindowForgetsOldCalls(t *testing.T) {
	cfg := Defaults()
	cfg.Window = 4
	// three identical failures long ago, then varied work: must not trigger
	msgs := turns(3, same("search", `{"q":"a"}`, "x"))
	more := turns(6, func(i int) (string, string, string) {
		return "open", fmt.Sprintf(`{"id":%d}`, i), fmt.Sprintf("page %d", i)
	})
	msgs = append(msgs, more[1:]...)
	if f := New(cfg).analyze(msgs, nil); len(f) != 0 {
		t.Fatalf("old calls outside the window should be forgotten: %+v", f)
	}
}

func TestPollReturningTheSameResultIsNotNoProgressWhenAllowed(t *testing.T) {
	cfg := Defaults()
	cfg.AllowRepeat = []string{"search"}
	msgs := turns(8, func(int) (string, string, string) { return "search", `{"q":"job"}`, `{"state":"running"}` })
	resp, _, notes := drive(t, cfg, msgs, replyCall("search", `{"q":"job"}`), nil)
	if len(resp.Message.ToolCalls) != 1 || len(notes) != 0 {
		t.Fatalf("an allowed poll was nudged or stopped: calls=%d notes=%v", len(resp.Message.ToolCalls), notes)
	}
	cfg.AllowRepeat = nil
	stuckMsgs := turns(8, func(int) (string, string, string) { return "search", `{"q":"job"}`, `{"hits":0}` })
	resp, _, _ = drive(t, cfg, stuckMsgs, replyCall("search", `{"q":"job"}`), nil)
	if len(resp.Message.ToolCalls) != 0 {
		t.Fatal("the same call with an unchanging non-progress result must still be caught")
	}
}

func TestCycleOfOnlyAllowedToolsIsExempt(t *testing.T) {
	cfg := Defaults()
	cfg.AllowRepeat = []string{"search", "fetch"}
	msgs := turns(12, func(i int) (string, string, string) {
		if i%2 == 0 {
			return "search", `{"q":"job"}`, fmt.Sprintf("r%d", i)
		}
		return "fetch", `{"id":"1"}`, fmt.Sprintf("r%d", i)
	})
	resp, _, notes := drive(t, cfg, msgs, replyCall("search", `{"q":"job"}`), nil)
	if len(resp.Message.ToolCalls) != 1 || len(notes) != 0 {
		t.Fatalf("a poll-and-wait cycle was treated as a loop: %v", notes)
	}
	cfg.AllowRepeat = []string{"search"} // fetch is not exempt, so the cycle is real
	resp, _, _ = drive(t, cfg, msgs, replyCall("search", `{"q":"job"}`), nil)
	if len(resp.Message.ToolCalls) != 0 {
		t.Fatal("a cycle that includes a non-exempt tool must still be caught")
	}
}

func TestInProgressResultsAreWaitingNotStuck(t *testing.T) {
	cfg := Defaults()
	poll := func(n int, result string) []core.Message {
		return turns(n, func(int) (string, string, string) { return "get_export", `{"id":"e-3"}`, result })
	}
	// no AllowRepeat list at all: the layer recognises a poll from what it returns
	for _, r := range []string{`{"state":"running"}`, `{"state":"queued"}`, "pending", `{"status":"in_progress","pct":40}`, "Job is still processing"} {
		resp, _, notes := drive(t, cfg, poll(12, r), replyCall("get_export", `{"id":"e-3"}`), nil)
		if len(resp.Message.ToolCalls) != 1 || len(notes) != 0 {
			t.Fatalf("a poll returning %q was nudged or stopped: %v", r, notes)
		}
	}
}

func TestAPollThatNeverFinishesIsStillStopped(t *testing.T) {
	cfg := Defaults()
	msgs := turns(45, func(int) (string, string, string) { return "get_export", `{"id":"e-3"}`, `{"state":"running"}` })
	resp, _, _ := drive(t, cfg, msgs, replyCall("get_export", `{"id":"e-3"}`), nil)
	if len(resp.Message.ToolCalls) != 0 {
		t.Fatal("45 identical polls that never finish should be stopped")
	}
}

func TestErrorsAreNeverTreatedAsProgress(t *testing.T) {
	cfg := Defaults()
	for _, r := range []string{`{"state":"pending","error":"503 try again"}`, "queue failed: still pending", "rate limited, request pending"} {
		msgs := turns(8, func(int) (string, string, string) { return "send", `{"to":"a"}`, r })
		resp, _, _ := drive(t, cfg, msgs, replyCall("send", `{"to":"a"}`), nil)
		if len(resp.Message.ToolCalls) != 0 {
			t.Fatalf("an error result %q was exempted as progress", r)
		}
	}
}

func TestReportsProgress(t *testing.T) {
	for r, want := range map[string]bool{
		`{"state":"running"}`: true, "Queued": true, `{"state":"done","rows":1842}`: false, `{"error":"429"}`: false,
		"completed": false, `{"state":"not ready"}`: true, "": false,
	} {
		if got := reportsProgress(r); got != want {
			t.Errorf("reportsProgress(%q) = %v, want %v", r, got, want)
		}
	}
}

func TestStreakOfEmptyResultsWithVaryingArgumentsGetsANudge(t *testing.T) {
	cfg := Defaults()
	queries := []string{"atlas plan", "atlas migration", "migration plan atlas", "plan for atlas migration", "atlas"}
	msgs := turns(5, func(i int) (string, string, string) {
		return "search", fmt.Sprintf(`{"q":%q}`, queries[i]), `{"hits":[],"total":0}`
	})
	_, p, notes := drive(t, cfg, msgs, replyCall("search", `{"q":"atlas doc"}`), nil)
	if len(notes) == 0 {
		t.Fatal("five empty searches with different arguments were not noticed")
	}
	sent := p.got[0].Messages
	if last := sent[len(sent)-1]; last.Role != core.RoleSystem || !strings.Contains(last.Content, "nothing usable") {
		t.Fatalf("the nudge did not name the problem: %+v", last)
	}
}

func TestLongStreakOfEmptyResultsIsStopped(t *testing.T) {
	cfg := Defaults()
	msgs := turns(9, func(i int) (string, string, string) {
		return "search", fmt.Sprintf(`{"q":"variant %d"}`, i), `{"results":[]}`
	})
	resp, _, _ := drive(t, cfg, msgs, replyCall("search", `{"q":"variant 10"}`), nil)
	if len(resp.Message.ToolCalls) != 0 || !strings.Contains(resp.Message.Content, "Stopped by Rakuvo") {
		t.Fatalf("a nine-call dry streak should halt: %+v", resp.Message)
	}
}

func TestOneUsefulResultEndsTheStreak(t *testing.T) {
	cfg := Defaults()
	msgs := turns(9, func(i int) (string, string, string) {
		if i == 5 {
			return "search", `{"q":"hit"}`, `{"hits":["doc-7"]}`
		}
		return "search", fmt.Sprintf(`{"q":"variant %d"}`, i), `{"hits":[]}`
	})
	resp, _, notes := drive(t, cfg, msgs, replyCall("search", `{"q":"another"}`), nil)
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatal("a streak broken by a useful result was stopped")
	}
	_ = notes
}

func TestAHealthyRetryAfterOneOrTwoEmptiesIsNotNudged(t *testing.T) {
	cfg := Defaults()
	msgs := turns(2, func(i int) (string, string, string) {
		return "search", fmt.Sprintf(`{"q":"variant %d"}`, i), `{"hits":[]}`
	})
	resp, _, notes := drive(t, cfg, msgs, replyCall("search", `{"q":"third"}`), nil)
	if len(resp.Message.ToolCalls) != 1 || len(notes) != 0 {
		t.Fatalf("two empty results are normal: %v", notes)
	}
}

func TestDryStreakIsPerToolAndHonoursAllowRepeat(t *testing.T) {
	cfg := Defaults()
	cfg.AllowRepeat = []string{"search"}
	msgs := turns(9, func(i int) (string, string, string) {
		return "search", fmt.Sprintf(`{"q":"variant %d"}`, i), `{"hits":[]}`
	})
	resp, _, _ := drive(t, cfg, msgs, replyCall("search", `{"q":"x"}`), nil)
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatal("an exempted tool was stopped for a dry streak")
	}
}

func TestUnproductive(t *testing.T) {
	for r, want := range map[string]bool{
		`{"hits":[]}`: true, `{"total":0}`: true, "0 results": true, "No records found": true, `{"error":"404 not found"}`: true,
		`{"results":[{"name":"Marta","phone":"555"}]}`: false, `{"state":"done","rows":1842}`: false, "ok": false,
	} {
		if got := unproductive(r); got != want {
			t.Errorf("unproductive(%q) = %v, want %v", r, got, want)
		}
	}
}
