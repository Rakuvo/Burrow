// Package contextmgr is the Context layer. It controls what the model reads on each
// call: it shrinks oversized tool results, blanks tool results that repeat a later one,
// and, when a token budget is set, drops the oldest whole turns until the request fits.
//
// Masking is the main tool: old tool results become one-line placeholders and the last
// few stay in full, which the research on agent context found as effective as
// summarizing at about half the cost. Truncating, deduplicating and dropping old turns
// are there for requests masking cannot shrink enough. Only the optional summarizer
// makes a model call; without it the layer adds no latency and every attempt of a
// request gets the same result. Three rules keep it safe:
//
//   - The last KeepRecentTurns turns are never changed, so the model always sees the
//     request it is answering and the tool results it is about to read.
//   - Turns are dropped whole, from a user message up to the next one, so a tool call
//     is never separated from its result. If a drop would still leave a tool result
//     without its call, the drop is abandoned.
//   - Leading system messages, any other system message, and the tool definitions are
//     never removed.
package contextmgr

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/rakuvo/burrow/core"
)

// Config selects what the layer does. The zero value changes nothing.
type Config struct {
	// MaxInputTokens is the token budget for the whole request, tool definitions
	// included. Over budget, the oldest unprotected turns are dropped. Zero turns the
	// budget off. It is an estimate (about 4 bytes per token), so leave headroom.
	MaxInputTokens int
	// MaxToolResultChars shortens an older tool result longer than this to its start
	// and end with a marker between. Zero turns it off.
	MaxToolResultChars int
	// DedupeMinChars replaces an older tool result that is identical to a later one,
	// and at least this long, with a short marker. Zero turns it off.
	DedupeMinChars int
	// KeepRecentTurns is how many of the latest turns are left untouched. A turn
	// starts at a user message. Values below 1 mean 1.
	KeepRecentTurns int

	// KeepLastObservations turns on observation masking: only this many of the latest
	// tool results stay in full, and every older one is replaced by a one-line
	// placeholder naming the tool, its arguments and the size of the output that was
	// left out. The tool call and the assistant's own words stay, so the model keeps
	// the story and can call the tool again if it needs the data. Zero turns it off.
	KeepLastObservations int
	// MaskStep makes masking advance in jumps. Providers cache the start of a prompt,
	// and masking one more result on every call would change the prompt early each
	// time. With a step of 4 the set of masked results only changes once four more
	// have become eligible, so the prompt is identical in between. Values below 2
	// mean no stepping.
	MaskStep int
	// MaskMinChars leaves results shorter than this alone, because a placeholder
	// would not be much shorter. Zero means 200.
	MaskMinChars int
	// MaskWithinTurn lets masking reach tool results inside the latest user turn. By default
	// the whole last turn is protected, which is right for chat, but a long agent run is often
	// one user request followed by dozens of tool calls: there the protected turn is the entire
	// history and nothing could ever be masked. The latest KeepLastObservations results still
	// stay in full either way.
	MaskWithinTurn bool
	// KeepRelevant keeps this many older tool results in full, besides the last
	// KeepLastObservations, chosen by how well they match the latest user message
	// (rare words shared with the request). Everything else old is masked, so a result
	// the user is asking about survives however old it is. Zero keeps only by age.
	KeepRelevant int
	// Facts, when set, writes a short note of the key values (IDs, names, numbers,
	// paths, errors) into each placeholder, so masking loses the bulk of an output but
	// not what a later step needs from it. It costs one small-model call per masked
	// result, made once and remembered.
	Facts FactExtractor

	// Summarizer, when set, turns the old turns that exceed the token budget into a
	// short summary instead of discarding them. The summary replaces them, so the
	// budget is the only thing that triggers it; masking runs first and is usually
	// enough. If the summarizer fails the turns are dropped as without it.
	Summarizer Summarizer
	// SummaryTokens is the room kept in the budget for the summary. Zero means 400.
	SummaryTokens int
}

func (c Config) summaryReserve() int {
	if c.SummaryTokens > 0 {
		return c.SummaryTokens
	}
	return 400
}

// Defaults is a reasonable starting point. The budget stays off because the right
// number depends on the model, so the caller sets it.
func Defaults() Config {
	return Config{MaxToolResultChars: 8000, DedupeMinChars: 200, KeepRecentTurns: 2}
}

// Masking is the configuration the research favours for agents: keep the last few
// observations, mask the rest in steps.
func Masking() Config {
	return Config{KeepLastObservations: 6, MaskStep: 3, KeepRecentTurns: 1}
}

// Layer implements core.Layer.
type Layer struct {
	cfg Config

	mu        sync.Mutex
	summaries map[string]string // transcript hash -> summary, so a repeated request is not summarized (and billed) again
	facts     map[string]string // tool call + output hash -> facts note, so a result is extracted once
}

func New(cfg Config) *Layer {
	if cfg.KeepRecentTurns < 1 {
		cfg.KeepRecentTurns = 1
	}
	return &Layer{cfg: cfg}
}

func (*Layer) Name() string { return "context" }

func (*Layer) After(context.Context, *core.Step) error { return nil }

const dedupeMarker = "[identical to a later tool result; omitted by Rakuvo]"

// Before rewrites the request. The engine gives every attempt a fresh copy of the
// caller's request, so nothing is ever trimmed twice.
func (l *Layer) Before(ctx context.Context, s *core.Step) error {
	cfg := l.cfg
	if cfg.MaxInputTokens <= 0 && cfg.MaxToolResultChars <= 0 && cfg.DedupeMinChars <= 0 && cfg.KeepLastObservations <= 0 {
		return nil
	}
	req := &s.Request
	if len(req.Messages) == 0 {
		return nil
	}
	startTokens := estimateTokens(*req)
	lead := leadingSystem(req.Messages)
	protect := protectedFrom(req.Messages, cfg.KeepRecentTurns)

	masked := l.mask(ctx, s, req.Messages, lead, protect)
	deduped := l.dedupe(req.Messages, lead, protect)
	truncated := l.truncate(req.Messages, lead, protect)
	dropped, abandoned := l.drop(ctx, s, lead, protect)

	if abandoned {
		s.Note(l.Name(), "dropping old turns would have separated a tool call from its result; left them in place")
	}
	if masked+deduped+truncated+dropped == 0 {
		return nil
	}
	var parts []string
	if masked > 0 {
		parts = append(parts, fmt.Sprintf("masked %d old tool result(s)", masked))
	}
	if deduped > 0 {
		parts = append(parts, fmt.Sprintf("blanked %d repeated tool result(s)", deduped))
	}
	if truncated > 0 {
		parts = append(parts, fmt.Sprintf("shortened %d long tool result(s)", truncated))
	}
	if dropped > 0 {
		parts = append(parts, fmt.Sprintf("dropped %d old turn(s)", dropped))
	}
	saved := startTokens - estimateTokens(*req)
	s.Note(l.Name(), fmt.Sprintf("%s; about %d input tokens saved", strings.Join(parts, ", "), saved))
	return nil
}

// mask replaces old tool results with placeholders. The last KeepLastObservations
// results stay in full; of the earlier ones, the oldest are masked, in multiples of
// MaskStep so the prompt only changes every MaskStep results. Results in the
// protected tail and short results are never masked.
func (l *Layer) mask(ctx context.Context, s *core.Step, msgs []core.Message, lead, protect int) int {
	keep := l.cfg.KeepLastObservations
	if keep <= 0 {
		return 0
	}
	var obs []int // indices of tool results, oldest first
	calls := map[string]core.ToolCall{}
	for i, m := range msgs {
		for _, c := range m.ToolCalls {
			calls[c.ID] = c
		}
		if i >= lead && m.Role == core.RoleTool {
			obs = append(obs, i)
		}
	}
	eligible := len(obs) - keep
	if step := l.cfg.MaskStep; step > 1 {
		eligible -= eligible % step
	}
	minChars := l.cfg.MaskMinChars
	if minChars <= 0 {
		minChars = 200
	}
	var todo []int
	for k := 0; k < eligible; k++ {
		i := obs[k]
		m := msgs[i]
		if (i >= protect && !l.cfg.MaskWithinTurn) || len(m.Content) < minChars || strings.HasPrefix(m.Content, maskPrefix) {
			continue
		}
		todo = append(todo, i)
	}
	if l.cfg.KeepRelevant > 0 {
		spared := relevant(msgs, todo, calls, l.cfg.KeepRelevant)
		kept := todo[:0]
		for _, i := range todo {
			if !spared[i] {
				kept = append(kept, i)
			}
		}
		todo = kept
	}
	facts := l.factsFor(ctx, s, msgs, todo, calls)
	for _, i := range todo {
		msgs[i].Content = placeholder(calls[msgs[i].ToolCallID], len(msgs[i].Content), facts[i])
	}
	return len(todo)
}

const maskPrefix = "[Rakuvo: output of "

// placeholder says what was left out and how to get it back. It is a pure function
// of the call and the size, so the same history always masks to the same text.
func placeholder(c core.ToolCall, size int, facts string) string {
	name := c.Name
	if name == "" {
		name = "a tool"
	}
	args := cutHead(strings.Join(strings.Fields(c.Arguments), " "), 100)
	if len(args) < len(strings.Join(strings.Fields(c.Arguments), " ")) {
		args += "..."
	}
	if facts != "" {
		return fmt.Sprintf("%s%s(%s) omitted, %d characters. Key facts: %s Call the tool again if you need the full data.]", maskPrefix, name, args, size, facts)
	}
	return fmt.Sprintf("%s%s(%s) omitted, %d characters. Call the tool again if you need this data.]", maskPrefix, name, args, size)
}

// dedupe blanks older tool results that match a later tool result. It scans from
// the end, so the copy that stays is the latest one.
func (l *Layer) dedupe(msgs []core.Message, lead, protect int) int {
	if l.cfg.DedupeMinChars <= 0 {
		return 0
	}
	seen := map[string]bool{}
	n := 0
	for i := len(msgs) - 1; i >= lead; i-- {
		m := &msgs[i]
		if m.Role != core.RoleTool || len(m.Content) < l.cfg.DedupeMinChars {
			continue
		}
		if seen[m.Content] && i < protect {
			m.Content = dedupeMarker
			n++
			continue
		}
		seen[m.Content] = true
	}
	return n
}

// truncate shortens older tool results that are too long, keeping the start and end.
func (l *Layer) truncate(msgs []core.Message, lead, protect int) int {
	limit := l.cfg.MaxToolResultChars
	if limit <= 0 {
		return 0
	}
	n := 0
	for i := lead; i < protect && i < len(msgs); i++ {
		m := &msgs[i]
		if m.Role != core.RoleTool || len(m.Content) <= limit {
			continue
		}
		head := cutHead(m.Content, limit/2)
		tail := cutTail(m.Content, limit-limit/2)
		omitted := len(m.Content) - len(head) - len(tail)
		marker := fmt.Sprintf("\n[... %d characters omitted by Rakuvo ...]\n", omitted)
		if omitted <= len(marker) {
			continue // not worth it
		}
		m.Content = head + marker + tail
		n++
	}
	return n
}

// drop removes the oldest unprotected turns until the request fits the budget. It
// returns how many turns it dropped, and whether it gave up to protect call/result pairs.
func (l *Layer) drop(ctx context.Context, s *core.Step, lead, protect int) (dropped int, abandoned bool) {
	req := &s.Request
	budget := l.cfg.MaxInputTokens
	if budget <= 0 || estimateTokens(*req) <= budget {
		return 0, false
	}
	if l.cfg.Summarizer != nil {
		budget -= l.cfg.summaryReserve() // leave room for the summary that replaces the turns
	}
	msgs := req.Messages
	starts := turnStarts(msgs, lead, protect)
	if len(starts) == 0 {
		return 0, false
	}
	orphansBefore := orphans(msgs)

	keep := make([]bool, len(msgs))
	for i := range keep {
		keep[i] = true
	}
	trial := *req
	for t, start := range starts {
		end := protect
		if t+1 < len(starts) {
			end = starts[t+1]
		}
		for i := start; i < end; i++ {
			if msgs[i].Role != core.RoleSystem { // later system messages stay
				keep[i] = false
			}
		}
		dropped++
		trial.Messages = filter(msgs, keep, lead, omitted(dropped))
		if estimateTokens(trial) <= budget {
			break
		}
	}
	if orphans(trial.Messages) > orphansBefore {
		return 0, true
	}
	if l.cfg.Summarizer != nil {
		var gone []core.Message
		for i := lead; i < protect; i++ {
			if !keep[i] {
				gone = append(gone, msgs[i])
			}
		}
		if text := l.summarize(ctx, s, gone); text != "" {
			trial.Messages = filter(msgs, keep, lead, summaryNote(dropped, text))
			s.Note(l.Name(), fmt.Sprintf("summarized %d old turn(s) into %d characters", dropped, len(text)))
		} // otherwise the plain omission note stays: a failing summarizer never blocks a request
	}
	req.Messages = trial.Messages
	return dropped, false
}

func omitted(turns int) string {
	return fmt.Sprintf("[Rakuvo: %d earlier turn(s) of this conversation were omitted to fit the context budget.]", turns)
}

func summaryNote(turns int, summary string) string {
	return fmt.Sprintf("[Rakuvo: %d earlier turn(s) of this conversation were replaced by this summary to fit the context budget.]\n%s", turns, summary)
}

// filter keeps the flagged messages and adds one note after the leading system messages.
func filter(msgs []core.Message, keep []bool, lead int, note string) []core.Message {
	out := make([]core.Message, 0, len(msgs)+1)
	out = append(out, msgs[:lead]...)
	out = append(out, core.Message{Role: core.RoleSystem, Content: note})
	for i := lead; i < len(msgs); i++ {
		if keep[i] {
			out = append(out, msgs[i])
		}
	}
	return out
}

// leadingSystem counts the system messages at the start of the conversation.
func leadingSystem(msgs []core.Message) int {
	n := 0
	for n < len(msgs) && msgs[n].Role == core.RoleSystem {
		n++
	}
	return n
}

// protectedFrom is the index where the protected tail begins: the start of the
// keep-th user message from the end. With fewer user turns than that, or none,
// everything is protected.
func protectedFrom(msgs []core.Message, keep int) int {
	var users []int
	for i, m := range msgs {
		if m.Role == core.RoleUser {
			users = append(users, i)
		}
	}
	if len(users) <= keep {
		return 0
	}
	return users[len(users)-keep]
}

// turnStarts lists where each unprotected turn begins. Messages between the leading
// system messages and the first user message form a turn of their own.
func turnStarts(msgs []core.Message, lead, protect int) []int {
	if protect <= lead {
		return nil
	}
	starts := []int{lead}
	for i := lead + 1; i < protect; i++ {
		if msgs[i].Role == core.RoleUser {
			starts = append(starts, i)
		}
	}
	return starts
}

// orphans counts tool results with no earlier call carrying their ID.
func orphans(msgs []core.Message) int {
	calls := map[string]bool{}
	n := 0
	for _, m := range msgs {
		for _, c := range m.ToolCalls {
			calls[c.ID] = true
		}
		if m.Role == core.RoleTool && !calls[m.ToolCallID] {
			n++
		}
	}
	return n
}

// estimateTokens is a rough count: about four bytes per token over every field the
// provider will be billed for. It is for budgeting, not billing.
func estimateTokens(req core.Request) int {
	bytes := 0
	for _, m := range req.Messages {
		bytes += len(m.Content) + 4
		for _, c := range m.ToolCalls {
			bytes += len(c.Name) + len(c.Arguments)
		}
	}
	for _, t := range req.Tools {
		bytes += len(t.Name) + len(t.Description) + len(t.Schema)
	}
	return (bytes + 3) / 4
}

// cutHead returns up to n bytes from the start of s, without splitting a character.
func cutHead(s string, n int) string {
	if n >= len(s) {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// cutTail returns up to n bytes from the end of s, without splitting a character.
func cutTail(s string, n int) string {
	if n >= len(s) {
		return s
	}
	i := len(s) - n
	for i < len(s) && !utf8.RuneStart(s[i]) {
		i++
	}
	return s[i:]
}
