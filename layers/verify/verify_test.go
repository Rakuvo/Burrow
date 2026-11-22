package verify_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/layers/prompt"
	"github.com/rakuvo/burrow/layers/verify"
	"github.com/rakuvo/burrow/providers/mock"
)

type fakeJudge struct {
	applies bool
	err     error
	calls   int
	seen    core.Request
}

func (f *fakeJudge) Judge(_ context.Context, req core.Request) (verify.Verdict, error) {
	f.calls++
	f.seen = req
	return verify.Verdict{Applies: f.applies, Usage: core.Usage{InputTokens: 100, OutputTokens: 2}}, f.err
}

var weather = core.ToolSpec{Name: "get_weather", Description: "Weather for a city", Schema: `{"type":"object"}`}

func callReply() mock.Reply {
	return mock.Reply{Response: &core.Response{Message: core.Message{Role: core.RoleAssistant,
		ToolCalls: []core.ToolCall{{ID: "1", Name: "get_weather", Arguments: `{}`}}}}}
}

func userReq() core.Request {
	return core.Request{Model: "m", Tools: []core.ToolSpec{weather}, Messages: []core.Message{{Role: core.RoleUser, Content: "weather in Pune?"}}}
}

func run(t *testing.T, j *fakeJudge, req core.Request, maxAttempts int, replies ...mock.Reply) (*core.Response, *mock.Provider, *core.MemoryTracer) {
	t.Helper()
	p, tr := mock.New(replies...), &core.MemoryTracer{}
	e := &core.Engine{Provider: p, Layers: []core.Layer{verify.New(j, verify.Config{Gate: true})}, Tracer: tr, MaxAttempts: maxAttempts}
	resp, err := e.Complete(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	return resp, p, tr
}

func TestSkippedToolIsRetriedWithACallRequired(t *testing.T) {
	j := &fakeJudge{applies: true}
	resp, p, tr := run(t, j, userReq(), 3, mock.Text("It is probably warm."), callReply())
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("resp = %+v", resp.Message)
	}
	reqs := p.Requests()
	if len(reqs) != 2 {
		t.Fatalf("attempts = %d", len(reqs))
	}
	if _, set := reqs[0].Extra["tool_choice"]; set {
		t.Fatal("the first attempt must be untouched")
	}
	if string(reqs[1].Extra["tool_choice"]) != `"required"` {
		t.Fatalf("retry must require a call: %v", reqs[1].Extra)
	}
	if j.calls != 1 {
		t.Fatalf("judge calls = %d, want 1 (the retry produced a call, so no more judging)", j.calls)
	}
	if n := tr.Events()[0].Notes; len(n) != 1 || !strings.Contains(n[0].Message, "a tool applies") {
		t.Fatalf("notes = %+v", n)
	}
}

func TestPlainAnswerIsKeptWhenNoToolApplies(t *testing.T) {
	j := &fakeJudge{applies: false}
	resp, p, _ := run(t, j, userReq(), 3, mock.Text("Paris."))
	if resp.Message.Content != "Paris." || len(p.Requests()) != 1 {
		t.Fatalf("a correct refusal to call must stand: %+v, attempts %d", resp.Message, len(p.Requests()))
	}
}

func TestJudgeIsNotConsultedWhenThereIsNothingToCatch(t *testing.T) {
	afterTool := userReq()
	afterTool.Messages = append(afterTool.Messages,
		core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "1", Name: "get_weather", Arguments: "{}"}}},
		core.Message{Role: core.RoleTool, ToolCallID: "1", Content: "31C"})
	noTools := userReq()
	noTools.Tools = nil
	forbidden := userReq()
	forbidden.Extra = map[string]json.RawMessage{"tool_choice": json.RawMessage(`"none"`)}
	required := userReq()
	required.Extra = map[string]json.RawMessage{"tool_choice": json.RawMessage(`"required"`)}
	forced := userReq()
	forced.Extra = map[string]json.RawMessage{"tool_choice": json.RawMessage(`{"type":"function","function":{"name":"get_weather"}}`)}

	for name, req := range map[string]core.Request{
		"after a tool result": afterTool, "no tools offered": noTools, "tool_choice none": forbidden,
		"tool_choice required": required, "tool_choice names a function": forced,
	} {
		j := &fakeJudge{applies: true}
		run(t, j, req, 3, mock.Text("final text"))
		if j.calls != 0 {
			t.Errorf("%s: the judge was called, which costs money and could override the caller", name)
		}
	}

	// A call was made: nothing to catch.
	j := &fakeJudge{applies: true}
	run(t, j, userReq(), 3, callReply())
	if j.calls != 0 {
		t.Error("the judge must not run when the model did call a tool")
	}

	// No attempt left to retry with: judging would be wasted spend.
	j = &fakeJudge{applies: true}
	run(t, j, userReq(), 1, mock.Text("text"))
	if j.calls != 0 {
		t.Error("the judge must not run when there is no retry left")
	}

	// A cut-off answer is not a skipped tool.
	cut := mock.Text("The weather in")
	cut.Response.FinishReason = "length"
	j = &fakeJudge{applies: true}
	run(t, j, userReq(), 3, cut)
	if j.calls != 0 {
		t.Error("a truncated answer is not evidence of a skipped call")
	}
}

func TestExplicitAutoStillGetsChecked(t *testing.T) {
	req := userReq()
	req.Extra = map[string]json.RawMessage{"tool_choice": json.RawMessage(`"auto"`)}
	j := &fakeJudge{applies: true}
	run(t, j, req, 3, mock.Text("text"), callReply())
	if j.calls != 1 {
		t.Fatalf("judge calls = %d", j.calls)
	}
}

func TestABrokenJudgeNeverBlocksAReply(t *testing.T) {
	j := &fakeJudge{err: errors.New("judge is down")}
	resp, p, tr := run(t, j, userReq(), 3, mock.Text("an answer"))
	if resp.Message.Content != "an answer" || len(p.Requests()) != 1 {
		t.Fatalf("a failing judge must fail open: %+v", resp.Message)
	}
	if n := tr.Events()[0].Notes; len(n) != 1 || !strings.Contains(n[0].Message, "judge unavailable") {
		t.Fatalf("the failure should be visible in the trace: %+v", n)
	}
}

func TestRetryStillTextIsAcceptedAfterTheCap(t *testing.T) {
	j := &fakeJudge{applies: true}
	resp, p, _ := run(t, j, userReq(), 2, mock.Text("first"), mock.Text("second"))
	if resp.Message.Content != "second" || len(p.Requests()) != 2 {
		t.Fatalf("the cap must end the loop: %+v attempts %d", resp.Message, len(p.Requests()))
	}
}

func TestCallersRequestIsNotMutated(t *testing.T) {
	req := userReq()
	run(t, &fakeJudge{applies: true}, req, 3, mock.Text("t"), callReply())
	if req.Extra != nil {
		t.Fatalf("the caller's request gained fields: %v", req.Extra)
	}
}

// ---------------------------------------------------------------- the judge

func judgeWith(reply string) (*verify.Judge, *mock.Provider) {
	p := mock.New(mock.Text(reply))
	return verify.NewJudge(p, "judge-model", "", 0), p
}

func TestJudgeParsesItsAnswers(t *testing.T) {
	for reply, want := range map[string]bool{"APPLIES": true, "applies.": true, " NONE\n": false, "None": false, `"APPLIES"`: true} {
		j, _ := judgeWith(reply)
		v, err := j.Judge(context.Background(), userReq())
		if err != nil || v.Applies != want {
			t.Errorf("reply %q: applies=%v err=%v, want %v", reply, v.Applies, err, want)
		}
	}
	j, _ := judgeWith("Hmm, it depends")
	if _, err := j.Judge(context.Background(), userReq()); err == nil {
		t.Error("an unusable answer must be an error, so the layer fails open")
	}
}

func TestJudgePromptCarriesOnlyWhatItNeeds(t *testing.T) {
	j, p := judgeWith("NONE")
	req := userReq()
	req.Messages = []core.Message{
		{Role: core.RoleSystem, Content: "SECRET SYSTEM PROMPT"},
		{Role: core.RoleUser, Content: "weather in Pune?"},
		{Role: core.RoleAssistant, Content: "ASSISTANT TEXT"},
	}
	req.Tools = []core.ToolSpec{{Name: "get_weather", Description: "Weather\nfor   a city", Schema: `{"secret":"schema"}`}}
	if _, err := j.Judge(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	sent := p.Requests()[0]
	body := sent.Messages[1].Content
	if sent.Model != "judge-model" || *sent.Temperature != 0 || sent.MaxTokens == 0 {
		t.Fatalf("judge request = %+v", sent)
	}
	for _, leak := range []string{"SECRET SYSTEM PROMPT", "ASSISTANT TEXT", "schema"} {
		if strings.Contains(body, leak) {
			t.Errorf("the judge was shown %q, which it does not need and the caller may not want shared", leak)
		}
	}
	if !strings.Contains(body, "weather in Pune?") || !strings.Contains(body, "- get_weather: Weather for a city") {
		t.Fatalf("judge input = %q", body)
	}
}

func TestJudgeBoundsItsInput(t *testing.T) {
	j, p := judgeWith("NONE")
	req := userReq()
	req.Messages[0].Content = strings.Repeat("x", 50000)
	for i := 0; i < 100; i++ {
		req.Tools = append(req.Tools, core.ToolSpec{Name: "t", Description: strings.Repeat("d", 5000)})
	}
	if _, err := j.Judge(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if n := len(p.Requests()[0].Messages[1].Content); n > 20000 {
		t.Fatalf("judge input is %d bytes: unbounded model- and user-controlled text", n)
	}
}

func TestJudgeProviderErrorIsReported(t *testing.T) {
	j := verify.NewJudge(mock.New(mock.Reply{Err: errors.New("boom")}), "m", "", 0)
	if _, err := j.Judge(context.Background(), userReq()); err == nil {
		t.Fatal("want an error")
	}
}

// ------------------------------------------------------------------- the veto

func vetoEngine(j verify.Judger, maxAttempts int, replies ...mock.Reply) (*core.Engine, *mock.Provider, *core.MemoryTracer) {
	p, tr := mock.New(replies...), &core.MemoryTracer{}
	return &core.Engine{
		Provider:    p,
		Layers:      []core.Layer{prompt.New(prompt.Config{}), verify.New(j, verify.Config{Veto: true})},
		Tracer:      tr,
		MaxAttempts: maxAttempts,
	}, p, tr
}

func TestVetoRetriesWithoutThePolicyWhenNoToolApplies(t *testing.T) {
	j := &fakeJudge{applies: false}
	e, p, tr := vetoEngine(j, 3, callReply(), mock.Text("Paris."))
	resp, err := e.Complete(context.Background(), userReq())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "Paris." {
		t.Fatalf("resp = %+v", resp.Message)
	}
	reqs := p.Requests()
	if len(reqs) != 2 {
		t.Fatalf("attempts = %d", len(reqs))
	}
	has := func(r core.Request) bool {
		for _, m := range r.Messages {
			if strings.Contains(m.Content, prompt.DefaultToolPolicy) {
				return true
			}
		}
		return false
	}
	if !has(reqs[0]) || has(reqs[1]) {
		t.Fatalf("policy on the first attempt only: first=%v retry=%v", has(reqs[0]), has(reqs[1]))
	}
	if j.calls != 1 {
		t.Fatalf("judge calls = %d, want 1: a vetoed request is never judged twice", j.calls)
	}
	if n := tr.Events()[0].Notes; len(n) != 2 || !strings.Contains(n[1].Message, "vetoed") {
		t.Fatalf("notes = %+v", n)
	}
}

func TestVetoKeepsACallTheJudgeAgreesWith(t *testing.T) {
	j := &fakeJudge{applies: true}
	e, p, _ := vetoEngine(j, 3, callReply())
	resp, err := e.Complete(context.Background(), userReq())
	if err != nil || len(resp.Message.ToolCalls) != 1 || len(p.Requests()) != 1 || j.calls != 1 {
		t.Fatalf("resp=%+v err=%v attempts=%d judge calls=%d", resp, err, len(p.Requests()), j.calls)
	}
}

func TestVetoOnlyAppliesToCallsThePolicyMayHaveCaused(t *testing.T) {
	// No Prompt layer: any call is the model's own choice and is left alone.
	j := &fakeJudge{applies: false}
	p := mock.New(callReply())
	e := &core.Engine{Provider: p, Layers: []core.Layer{verify.New(j, verify.Config{Veto: true})}}
	if _, err := e.Complete(context.Background(), userReq()); err != nil {
		t.Fatal(err)
	}
	if j.calls != 0 {
		t.Fatal("without the policy, a call is the model's own and must not be vetoed")
	}
}

func TestVetoSkipsWhatItShouldNotTouch(t *testing.T) {
	afterTool := userReq()
	afterTool.Messages = append(afterTool.Messages,
		core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "1", Name: "get_weather", Arguments: "{}"}}},
		core.Message{Role: core.RoleTool, ToolCallID: "1", Content: "31C"})
	forced := userReq()
	forced.Extra = map[string]json.RawMessage{"tool_choice": json.RawMessage(`"required"`)}
	for name, req := range map[string]core.Request{"after a tool result": afterTool, "tool_choice required": forced} {
		j := &fakeJudge{applies: false}
		e, _, _ := vetoEngine(j, 3, callReply())
		if _, err := e.Complete(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if j.calls != 0 {
			t.Errorf("%s: the judge was consulted", name)
		}
	}
	j := &fakeJudge{applies: false}
	e, _, _ := vetoEngine(j, 1, callReply())
	if _, err := e.Complete(context.Background(), userReq()); err != nil || j.calls != 0 {
		t.Errorf("no retry left: judge calls = %d err = %v", j.calls, err)
	}
}

func TestVetoFailsOpen(t *testing.T) {
	j := &fakeJudge{err: errors.New("judge is down")}
	e, p, _ := vetoEngine(j, 3, callReply())
	resp, err := e.Complete(context.Background(), userReq())
	if err != nil || len(resp.Message.ToolCalls) != 1 || len(p.Requests()) != 1 {
		t.Fatalf("a failing judge must not change the response: %+v %v", resp, err)
	}
}

func TestVetoedRetryThatCallsAgainIsAccepted(t *testing.T) {
	j := &fakeJudge{applies: false}
	e, p, _ := vetoEngine(j, 3, callReply(), callReply())
	resp, err := e.Complete(context.Background(), userReq())
	if err != nil || len(resp.Message.ToolCalls) != 1 || len(p.Requests()) != 2 || j.calls != 1 {
		t.Fatalf("one veto per request, then the model's own choice stands: %+v %v attempts=%d judge=%d", resp, err, len(p.Requests()), j.calls)
	}
}

func TestZeroConfigDoesNothing(t *testing.T) {
	j := &fakeJudge{applies: true}
	p := mock.New(mock.Text("text"))
	e := &core.Engine{Provider: p, Layers: []core.Layer{verify.New(j, verify.Config{})}}
	if _, err := e.Complete(context.Background(), userReq()); err != nil || j.calls != 0 {
		t.Fatalf("an unconfigured layer must be inert: judge calls = %d", j.calls)
	}
}

func TestJudgeSpendIsReportedSeparatelyFromTheModelsOwn(t *testing.T) {
	j := &fakeJudge{applies: false}
	e, _, tr := vetoEngine(j, 3, callReply(), mock.Text("Paris."))
	resp, err := e.Complete(context.Background(), userReq())
	if err != nil {
		t.Fatal(err)
	}
	ev := tr.Events()
	if ev[0].AuxUsage != (core.Usage{InputTokens: 100, OutputTokens: 2}) {
		t.Fatalf("attempt 1 aux usage = %+v: the judge ran and must be counted", ev[0].AuxUsage)
	}
	if ev[1].AuxUsage != (core.Usage{}) {
		t.Fatalf("attempt 2 aux usage = %+v: the judge did not run on the retry", ev[1].AuxUsage)
	}
	// callReply reports no usage and mock.Text reports 10 in / 5 out, so the
	// supervised model's own total is exactly that; the judge's 100 must not be in it.
	if resp.Usage != (core.Usage{InputTokens: 10, OutputTokens: 5}) {
		t.Fatalf("returned usage must be the supervised model's own, got %+v", resp.Usage)
	}
}

func TestFailedJudgeCallsAreStillCounted(t *testing.T) {
	j := &fakeJudge{err: errors.New("down")}
	e, _, tr := vetoEngine(j, 3, callReply())
	if _, err := e.Complete(context.Background(), userReq()); err != nil {
		t.Fatal(err)
	}
	if tr.Events()[0].AuxUsage.InputTokens != 100 {
		t.Fatalf("aux usage = %+v", tr.Events()[0].AuxUsage)
	}
}

func TestSingleRequestOnlySkipsAgentSessions(t *testing.T) {
	multi := userReq()
	multi.Messages = append([]core.Message{{Role: core.RoleUser, Content: "hi"}, {Role: core.RoleAssistant, Content: "hello"}}, multi.Messages...)
	for name, tc := range map[string]struct {
		req   core.Request
		calls int
	}{"one request": {userReq(), 1}, "later turn": {multi, 0}} {
		j := &fakeJudge{applies: true}
		p, tr := mock.New(mock.Reply{Response: &core.Response{Message: core.Message{Role: core.RoleAssistant, Content: "text"}}}, callReply()), &core.MemoryTracer{}
		e := &core.Engine{Provider: p, Layers: []core.Layer{verify.New(j, verify.Config{Gate: true, SingleRequestOnly: true})}, Tracer: tr, MaxAttempts: 2}
		if _, err := e.Complete(context.Background(), tc.req); err != nil {
			t.Fatal(err)
		}
		if j.calls != tc.calls {
			t.Errorf("%s: judge consulted %d times, want %d", name, j.calls, tc.calls)
		}
	}
}

func TestJudgeCachesRepeatedRequests(t *testing.T) {
	prov := mock.New(
		mock.Reply{Response: &core.Response{Message: core.Message{Role: core.RoleAssistant, Content: "APPLIES"}, Usage: core.Usage{InputTokens: 50}}},
	)
	j := verify.NewJudge(prov, "j", "", 0)
	for i := 0; i < 3; i++ {
		v, err := j.Judge(context.Background(), userReq())
		if err != nil || !v.Applies {
			t.Fatalf("call %d: %v %+v", i, err, v)
		}
		if i > 0 && v.Usage.InputTokens != 0 {
			t.Fatalf("cached verdict reported spend: %+v", v.Usage)
		}
	}
}
