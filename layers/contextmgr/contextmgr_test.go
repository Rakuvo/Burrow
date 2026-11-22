package contextmgr

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/rakuvo/burrow/core"
)

func run(t *testing.T, cfg Config, req core.Request) (core.Request, *core.Step) {
	t.Helper()
	s := &core.Step{Original: req.Clone(), Request: req.Clone(), State: map[string]any{}, Attempt: 1, MaxAttempts: 3}
	if err := New(cfg).Before(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	return s.Request, s
}

func user(c string) core.Message { return core.Message{Role: core.RoleUser, Content: c} }
func sys(c string) core.Message  { return core.Message{Role: core.RoleSystem, Content: c} }

func call(id, name string) core.Message {
	return core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: id, Name: name, Arguments: "{}"}}}
}

func result(id, c string) core.Message {
	return core.Message{Role: core.RoleTool, ToolCallID: id, Content: c}
}

// conversation builds n turns, each: user, assistant call, tool result of the given size.
func conversation(n, resultSize int) []core.Message {
	msgs := []core.Message{sys("be helpful")}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("c%d", i)
		msgs = append(msgs, user(fmt.Sprintf("question %d", i)), call(id, "lookup"),
			result(id, fmt.Sprintf("%d:", i)+strings.Repeat("x", resultSize)),
			core.Message{Role: core.RoleAssistant, Content: fmt.Sprintf("answer %d", i)})
	}
	return msgs
}

// traced runs the layer inside the engine, because notes only reach the outside through trace events.
func traced(t *testing.T, cfg Config, req core.Request) (core.Request, []string) {
	t.Helper()
	tr := &core.MemoryTracer{}
	cap := &capture{}
	e := &core.Engine{Provider: cap, Layers: []core.Layer{New(cfg)}, Tracer: tr}
	if _, err := e.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	var notes []string
	for _, ev := range tr.Events() {
		for _, n := range ev.Notes {
			notes = append(notes, n.Message)
		}
	}
	return cap.got, notes
}

// capture is a provider that records the request it was sent.
type capture struct{ got core.Request }

func (*capture) Name() string { return "capture" }
func (c *capture) Complete(_ context.Context, r core.Request) (*core.Response, error) {
	c.got = r
	return &core.Response{}, nil
}

func TestZeroConfigChangesNothing(t *testing.T) {
	req := core.Request{Model: "m", Messages: conversation(6, 5000)}
	got, _ := run(t, Config{}, req)
	if fmt.Sprint(got.Messages) != fmt.Sprint(req.Messages) {
		t.Fatal("zero config changed the request")
	}
}

func TestTruncatesOldToolResultsOnly(t *testing.T) {
	req := core.Request{Messages: conversation(4, 5000)}
	got, _ := run(t, Config{MaxToolResultChars: 1000, KeepRecentTurns: 2}, req)
	// turns 0 and 1 are old; turns 2 and 3 are protected
	for i, m := range got.Messages {
		if m.Role != core.RoleTool {
			continue
		}
		old := i < 9 // sys + 2 full turns of 4 messages = indices 0..8
		if old && len(m.Content) > 1100 {
			t.Fatalf("old result %d not shortened (%d bytes)", i, len(m.Content))
		}
		if !old && len(m.Content) < 5000 {
			t.Fatalf("recent result %d was changed (%d bytes)", i, len(m.Content))
		}
	}
	if !strings.Contains(got.Messages[3].Content, "characters omitted by Rakuvo") {
		t.Fatal("no marker in shortened result")
	}
	if !strings.HasPrefix(got.Messages[3].Content, "0:xxx") || !strings.HasSuffix(got.Messages[3].Content, "xxx") {
		t.Fatal("start and end of the result should be kept")
	}
}

func TestTruncateKeepsValidUTF8(t *testing.T) {
	big := strings.Repeat("日本語", 2000)
	req := core.Request{Messages: []core.Message{user("q1"), call("a", "f"), result("a", big), user("q2"), user("q3")}}
	got, _ := run(t, Config{MaxToolResultChars: 1001, KeepRecentTurns: 2}, req)
	if !utf8.ValidString(got.Messages[2].Content) {
		t.Fatal("truncation split a character")
	}
	if len(got.Messages[2].Content) >= len(big) {
		t.Fatal("not shortened")
	}
}

func TestDedupeKeepsLatestCopy(t *testing.T) {
	same := strings.Repeat("same data ", 50)
	req := core.Request{Messages: []core.Message{
		user("q1"), call("a", "f"), result("a", same),
		user("q2"), call("b", "f"), result("b", same),
		user("q3"), call("c", "f"), result("c", same),
		user("q4"),
	}}
	got, _ := run(t, Config{DedupeMinChars: 100, KeepRecentTurns: 3}, req) // q2, q3, q4 are protected
	if got.Messages[2].Content != dedupeMarker {
		t.Fatal("oldest copy should be blanked")
	}
	if got.Messages[5].Content != same || got.Messages[8].Content != same {
		t.Fatal("copies in the protected turns must be kept")
	}
	if got.Messages[2].ToolCallID != "a" || got.Messages[2].Role != core.RoleTool {
		t.Fatal("the message itself must stay, so the call keeps its result")
	}
}

func TestDedupeIgnoresShortResults(t *testing.T) {
	req := core.Request{Messages: []core.Message{
		user("q1"), call("a", "f"), result("a", "ok"),
		user("q2"), call("b", "f"), result("b", "ok"),
		user("q3"), user("q4"),
	}}
	got, _ := run(t, Config{DedupeMinChars: 100, KeepRecentTurns: 1}, req)
	if got.Messages[2].Content != "ok" {
		t.Fatal("short results must not be blanked")
	}
}

func TestBudgetDropsOldestWholeTurns(t *testing.T) {
	req := core.Request{Messages: conversation(10, 4000)}
	before := estimateTokens(req)
	budget := before / 2
	got, _ := run(t, Config{MaxInputTokens: budget, KeepRecentTurns: 2}, req)
	if est := estimateTokens(got); est > budget {
		t.Fatalf("still over budget: %d > %d", est, budget)
	}
	if got.Messages[0].Role != core.RoleSystem || got.Messages[0].Content != "be helpful" {
		t.Fatal("leading system message must stay first")
	}
	if got.Messages[1].Role != core.RoleSystem || !strings.Contains(got.Messages[1].Content, "omitted") {
		t.Fatalf("expected an omission marker, got %+v", got.Messages[1])
	}
	if o := orphans(got.Messages); o != 0 {
		t.Fatalf("%d tool results lost their call", o)
	}
	last := got.Messages[len(got.Messages)-1]
	if last.Content != "answer 9" {
		t.Fatalf("latest turn must be intact, last = %+v", last)
	}
	// the two protected turns are fully there
	if !strings.Contains(fmt.Sprint(got.Messages), "question 8") || !strings.Contains(fmt.Sprint(got.Messages), "question 9") {
		t.Fatal("protected turns missing")
	}
	if strings.Contains(fmt.Sprint(got.Messages), "question 0") {
		t.Fatal("oldest turn should have been dropped")
	}
}

func TestBudgetNeverTouchesProtectedTurns(t *testing.T) {
	req := core.Request{Messages: conversation(3, 4000)}
	got, _ := run(t, Config{MaxInputTokens: 10, KeepRecentTurns: 3}, req) // impossible budget
	if fmt.Sprint(got.Messages) != fmt.Sprint(req.Messages) {
		t.Fatal("with every turn protected nothing may change")
	}
}

func TestToolDefinitionsCountTowardBudget(t *testing.T) {
	req := core.Request{
		Messages: conversation(4, 100),
		Tools:    []core.ToolSpec{{Name: "lookup", Description: strings.Repeat("d", 4000), Schema: "{}"}},
	}
	withMsgsOnly := estimateTokens(core.Request{Messages: req.Messages})
	if estimateTokens(req) < withMsgsOnly+900 {
		t.Fatal("tool definitions were not counted")
	}
	got, _ := run(t, Config{MaxInputTokens: withMsgsOnly, KeepRecentTurns: 1}, req)
	if len(got.Tools) != 1 || len(got.Tools[0].Description) != 4000 {
		t.Fatal("tools must never be changed")
	}
	if len(got.Messages) >= len(req.Messages)+1 {
		t.Fatal("budget should have forced turns out because of the tools")
	}
}

func TestDropAbandonedWhenItWouldOrphanAResult(t *testing.T) {
	// A malformed history: the result for call "x" arrives after the next user message,
	// so dropping the first turn alone would strand it.
	req := core.Request{Messages: []core.Message{
		user(strings.Repeat("q", 8000)), call("x", "f"),
		user("q2"), result("x", "small"),
		user("q3"), user("q4"),
	}}
	got, notes := traced(t, Config{MaxInputTokens: 600, KeepRecentTurns: 1}, req)
	if fmt.Sprint(got.Messages) != fmt.Sprint(req.Messages) {
		t.Fatal("the drop should have been abandoned and the history left as sent")
	}
	if len(notes) != 1 || !strings.Contains(notes[0], "separated a tool call from its result") {
		t.Fatalf("expected an abandon note, got %v", notes)
	}
}

func TestLaterSystemMessagesSurviveDrop(t *testing.T) {
	msgs := conversation(6, 3000)
	// a rule injected between turns 0 and 1 (index 5 is turn 1's user message)
	msgs = append(msgs[:5], append([]core.Message{sys("mid-conversation rule")}, msgs[5:]...)...)
	req := core.Request{Messages: msgs}
	got, _ := run(t, Config{MaxInputTokens: estimateTokens(req) / 3, KeepRecentTurns: 1}, req)
	if !strings.Contains(fmt.Sprint(got.Messages), "mid-conversation rule") {
		t.Fatal("a system message inside a dropped turn must be kept")
	}
}

func TestIdempotentAcrossAttempts(t *testing.T) {
	cfg := Config{MaxInputTokens: 3000, MaxToolResultChars: 500, DedupeMinChars: 100, KeepRecentTurns: 2}
	req := core.Request{Messages: conversation(8, 3000)}
	a, _ := run(t, cfg, req)
	b, _ := run(t, cfg, req)
	if fmt.Sprint(a.Messages) != fmt.Sprint(b.Messages) {
		t.Fatal("same input must give the same output")
	}
}

func TestDoesNotMutateCallersRequest(t *testing.T) {
	req := core.Request{Messages: conversation(6, 3000)}
	orig := req.Clone()
	s := &core.Step{Original: req.Clone(), Request: req.Clone(), State: map[string]any{}}
	_ = New(Config{MaxInputTokens: 2000, MaxToolResultChars: 300, KeepRecentTurns: 1}).Before(context.Background(), s)
	if fmt.Sprint(s.Original.Messages) != fmt.Sprint(orig.Messages) {
		t.Fatal("Original was modified")
	}
}

func TestRecordsWhatItDidInTheTrace(t *testing.T) {
	_, notes := traced(t, Config{MaxToolResultChars: 500, KeepRecentTurns: 1}, core.Request{Model: "m", Messages: conversation(4, 3000)})
	if len(notes) != 1 || !strings.Contains(notes[0], "shortened 3 long tool result") || !strings.Contains(notes[0], "tokens saved") {
		t.Fatalf("unexpected notes %v", notes)
	}
}

func toolResults(msgs []core.Message) []string {
	var out []string
	for _, m := range msgs {
		if m.Role == core.RoleTool {
			out = append(out, m.Content)
		}
	}
	return out
}

func masked(c string) bool { return strings.HasPrefix(c, maskPrefix) }

func TestMaskKeepsLastNObservationsInFull(t *testing.T) {
	req := core.Request{Messages: conversation(10, 2000)}
	got, notes := traced(t, Config{KeepLastObservations: 3, KeepRecentTurns: 1}, req)
	res := toolResults(got.Messages)
	if len(res) != 10 {
		t.Fatalf("every tool result message must stay, got %d", len(res))
	}
	for i, c := range res {
		if want := i < 7; masked(c) != want {
			t.Fatalf("result %d masked=%v, want %v", i, masked(c), want)
		}
	}
	if len(notes) != 1 || !strings.Contains(notes[0], "masked 7 old tool result") {
		t.Fatalf("notes: %v", notes)
	}
}

func TestMaskPlaceholderNamesToolArgsAndSize(t *testing.T) {
	msgs := []core.Message{
		user("q1"),
		{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "a", Name: "get_weather", Arguments: "{\"city\":  \"Paris\"}"}}},
		result("a", strings.Repeat("w", 900)),
		user("q2"), user("q3"),
	}
	got, _ := run(t, Config{KeepLastObservations: 1, KeepRecentTurns: 1}, core.Request{Messages: msgs})
	// only one observation and it is the last one, so it stays
	if masked(got.Messages[2].Content) {
		t.Fatal("the only observation is the newest and must stay")
	}
	msgs = append(msgs[:3:3], core.Message{Role: core.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "b", Name: "f", Arguments: "{}"}}}, result("b", strings.Repeat("z", 500)), user("q4"))
	got, _ = run(t, Config{KeepLastObservations: 1, KeepRecentTurns: 1}, core.Request{Messages: msgs})
	p := got.Messages[2].Content
	for _, want := range []string{"get_weather", `{"city": "Paris"}`, "900 characters", "Call the tool again"} {
		if !strings.Contains(p, want) {
			t.Fatalf("placeholder %q lacks %q", p, want)
		}
	}
	if got.Messages[2].ToolCallID != "a" || got.Messages[1].ToolCalls[0].ID != "a" {
		t.Fatal("the call and its result must stay paired")
	}
}

func TestMaskStepOnlyChangesInJumps(t *testing.T) {
	cfg := Config{KeepLastObservations: 2, MaskStep: 4, KeepRecentTurns: 1}
	count := func(n int) int {
		got, _ := run(t, cfg, core.Request{Messages: conversation(n, 1000)})
		c := 0
		for _, r := range toolResults(got.Messages) {
			if masked(r) {
				c++
			}
		}
		return c
	}
	// eligible = n-2; masked = eligible rounded down to a multiple of 4
	for n, want := range map[int]int{3: 0, 5: 0, 6: 4, 7: 4, 9: 4, 10: 8, 11: 8} {
		if got := count(n); got != want {
			t.Fatalf("with %d results %d masked, want %d", n, got, want)
		}
	}
}

func TestMaskPrefixIsStableAsConversationGrows(t *testing.T) {
	cfg := Config{KeepLastObservations: 2, MaskStep: 4, KeepRecentTurns: 1}
	a, _ := run(t, cfg, core.Request{Messages: conversation(6, 1000)})
	b, _ := run(t, cfg, core.Request{Messages: conversation(9, 1000)})
	// the first 4 turns (sys + 4*4 messages) must be byte-identical between the two requests
	if fmt.Sprint(a.Messages[:17]) != fmt.Sprint(b.Messages[:17]) {
		t.Fatal("the cached prefix changed between steps")
	}
}

func TestMaskSkipsShortResultsAndIsIdempotent(t *testing.T) {
	req := core.Request{Messages: conversation(6, 50)}
	got, _ := run(t, Config{KeepLastObservations: 1, KeepRecentTurns: 1}, req)
	for _, r := range toolResults(got.Messages) {
		if masked(r) {
			t.Fatal("short results must be left alone")
		}
	}
	long := core.Request{Messages: conversation(6, 1000)}
	once, _ := run(t, Config{KeepLastObservations: 1, KeepRecentTurns: 1}, long)
	twice, _ := run(t, Config{KeepLastObservations: 1, KeepRecentTurns: 1}, once)
	if fmt.Sprint(once.Messages) != fmt.Sprint(twice.Messages) {
		t.Fatal("masking an already-masked history must change nothing")
	}
}

func TestMaskSavesTokens(t *testing.T) {
	req := core.Request{Messages: conversation(12, 4000)}
	got, notes := traced(t, Masking(), req)
	// 12 results, 6 kept in full: about half the tokens should go
	if estimateTokens(got) > estimateTokens(req)*6/10 {
		t.Fatalf("expected about half the tokens saved, %d -> %d", estimateTokens(req), estimateTokens(got))
	}
	if len(notes) != 1 || !strings.Contains(notes[0], "tokens saved") {
		t.Fatalf("notes: %v", notes)
	}
}

type fakeSummarizer struct {
	calls int
	out   string
	err   error
	seen  string
}

func (f *fakeSummarizer) Summarize(_ context.Context, tr string) (string, core.Usage, error) {
	f.calls++
	f.seen = tr
	return f.out, core.Usage{InputTokens: 100, OutputTokens: 20}, f.err
}

func TestSummarizerReplacesDroppedTurns(t *testing.T) {
	fs := &fakeSummarizer{out: "user wants Paris weather; city id 42"}
	req := core.Request{Messages: conversation(10, 4000)}
	cfg := Config{MaxInputTokens: estimateTokens(req) / 2, KeepRecentTurns: 2, Summarizer: fs, SummaryTokens: 100}
	got, notes := traced(t, cfg, req)
	if fs.calls != 1 {
		t.Fatalf("summarizer called %d times", fs.calls)
	}
	if !strings.Contains(fs.seen, "question 0") || !strings.Contains(fs.seen, "tool result:") {
		t.Fatalf("summarizer should see the dropped turns, saw %q", fs.seen[:60])
	}
	if strings.Contains(fs.seen, "question 9") {
		t.Fatal("protected turns must not be sent to the summarizer")
	}
	if !strings.Contains(fmt.Sprint(got.Messages[1]), "city id 42") || !strings.Contains(got.Messages[1].Content, "replaced by this summary") {
		t.Fatalf("summary missing: %+v", got.Messages[1])
	}
	if estimateTokens(got) > estimateTokens(req)/2 {
		t.Fatal("request still over budget with the summary in")
	}
	var found bool
	for _, n := range notes {
		found = found || strings.Contains(n, "summarized")
	}
	if !found {
		t.Fatalf("no summarize note in %v", notes)
	}
}

func TestSummarizerFailureFallsBackToDropping(t *testing.T) {
	fs := &fakeSummarizer{err: fmt.Errorf("down")}
	req := core.Request{Messages: conversation(10, 4000)}
	got, notes := traced(t, Config{MaxInputTokens: estimateTokens(req) / 2, KeepRecentTurns: 2, Summarizer: fs}, req)
	if !strings.Contains(got.Messages[1].Content, "were omitted") {
		t.Fatalf("expected the plain omission note, got %q", got.Messages[1].Content)
	}
	if orphans(got.Messages) != 0 || got.Messages[len(got.Messages)-1].Content != "answer 9" {
		t.Fatal("request must still be valid")
	}
	if !strings.Contains(strings.Join(notes, "|"), "summarizer unavailable") {
		t.Fatalf("notes: %v", notes)
	}
}

func TestSummarizerResultIsCached(t *testing.T) {
	fs := &fakeSummarizer{out: "s"}
	l := New(Config{MaxInputTokens: 1, KeepRecentTurns: 1, Summarizer: fs})
	gone := conversation(3, 100)
	s := &core.Step{State: map[string]any{}}
	l.summarize(context.Background(), s, gone)
	l.summarize(context.Background(), s, gone)
	if fs.calls != 1 {
		t.Fatalf("second identical transcript should come from the cache, calls=%d", fs.calls)
	}
	if s.AuxUsage.InputTokens != 100 {
		t.Fatalf("only the first call is billed, aux input = %d", s.AuxUsage.InputTokens)
	}
}

func TestSummarizerNotUsedWhenMaskingIsEnough(t *testing.T) {
	fs := &fakeSummarizer{out: "s"}
	req := core.Request{Messages: conversation(12, 4000)}
	cfg := Masking()
	cfg.MaxInputTokens = estimateTokens(req) * 7 / 10 // masking alone gets to about 50%
	cfg.Summarizer = fs
	got, _ := traced(t, cfg, req)
	if fs.calls != 0 {
		t.Fatal("summarizer must only run when masking leaves the request over budget")
	}
	if estimateTokens(got) > cfg.MaxInputTokens {
		t.Fatal("over budget")
	}
}

func TestSummarizerCostIsReportedAsAuxUsage(t *testing.T) {
	fs := &fakeSummarizer{out: "s"}
	req := core.Request{Messages: conversation(10, 4000)}
	tr := &core.MemoryTracer{}
	e := &core.Engine{Provider: &capture{}, Layers: []core.Layer{New(Config{MaxInputTokens: estimateTokens(req) / 2, KeepRecentTurns: 2, Summarizer: fs})}, Tracer: tr}
	if _, err := e.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if ev := tr.Events()[0]; ev.AuxUsage.InputTokens != 100 || ev.AuxUsage.OutputTokens != 20 {
		t.Fatalf("aux usage = %+v", ev.AuxUsage)
	}
}

func TestShortConversationsAreUntouched(t *testing.T) {
	req := core.Request{Messages: []core.Message{sys("s"), user("hi")}}
	got, _ := run(t, Defaults(), req)
	if fmt.Sprint(got.Messages) != fmt.Sprint(req.Messages) {
		t.Fatal("changed a short conversation")
	}
}

func TestEmptyRequest(t *testing.T) {
	got, _ := run(t, Defaults(), core.Request{Model: "m"})
	if len(got.Messages) != 0 {
		t.Fatal("an empty request must stay empty")
	}
}

// A long agent run is one user request followed by many tool calls; the whole history is
// the "last turn", which is protected by default. MaskWithinTurn lets masking reach into it.
func agentRun(n int) []core.Message {
	msgs := []core.Message{sys("be helpful"), user("read all the pages")}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("c%d", i)
		msgs = append(msgs, call(id, "fetch_page"), result(id, fmt.Sprintf("%d:", i)+strings.Repeat("x", 500)))
	}
	return msgs
}

func countMasked(msgs []core.Message) int {
	n := 0
	for _, m := range msgs {
		if m.Role == core.RoleTool && strings.HasPrefix(m.Content, maskPrefix) {
			n++
		}
	}
	return n
}

func TestSingleRequestAgentRunIsNotMaskedByDefault(t *testing.T) {
	got, _ := run(t, Config{KeepLastObservations: 3, KeepRecentTurns: 1}, core.Request{Messages: agentRun(8)})
	if n := countMasked(got.Messages); n != 0 {
		t.Fatalf("masked %d results inside the protected turn, want 0", n)
	}
}

func TestMaskWithinTurnMasksOldResultsOfALongRunAndKeepsTheLatest(t *testing.T) {
	got, _ := run(t, Config{KeepLastObservations: 3, KeepRecentTurns: 1, MaskWithinTurn: true}, core.Request{Messages: agentRun(8)})
	if n := countMasked(got.Messages); n != 5 {
		t.Fatalf("masked %d results, want 5 (8 minus the 3 kept)", n)
	}
	var tools []core.Message
	for _, m := range got.Messages {
		if m.Role == core.RoleTool {
			tools = append(tools, m)
		}
	}
	for i, m := range tools[len(tools)-3:] {
		if strings.HasPrefix(m.Content, maskPrefix) {
			t.Fatalf("one of the latest three results was masked: %d %q", i, m.Content[:30])
		}
	}
}
