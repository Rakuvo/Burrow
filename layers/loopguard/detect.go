package loopguard

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/rakuvo/burrow/core"
)

// Level says how serious a finding is.
type Level int

const (
	Warn Level = iota + 1 // nudge the model
	Stop                  // halt the loop
)

// Finding is one detector's verdict.
type Finding struct {
	Kind   string // "no-progress", "repeat", "cycle", "text", "limit"
	Level  Level
	Count  int
	Tool   string
	Detail string
}

// entry is one tool call in the history, with a hash of the result it got, if any.
type entry struct {
	fp       string
	name     string
	result   string // hash of the result content; empty if no result is present
	progress bool   // the result says the work is still in progress (a running job, a queued task)
	dry      bool   // the result is empty or an error: the call produced nothing usable
}

// emptyRe matches results that say nothing was found.
var emptyRe = regexp.MustCompile(`(?i)(\[\s*\]|"total"\s*:\s*0\b|\b0 (results|hits|matches|records|items)\b|\bno (results|data|matches|records|hits|items)\b|zero.?results|nothing found|none found|"results"\s*:\s*\[\s*\]|"hits"\s*:\s*\[\s*\])`)

// unproductive reports whether a result gave the agent nothing to work with: an error, or an
// empty answer. A single one is normal; a streak of them from one tool means the agent is
// guessing at variations of a request that cannot be satisfied.
func unproductive(content string) bool {
	if len(content) > 2000 {
		content = content[:2000]
	}
	return errorRe.MatchString(content) || emptyRe.MatchString(content)
}

// A result that reports work in progress is a poll answering "not yet", not a call that
// cannot succeed. Results that mention an error are never treated this way.
var inProgressRe = regexp.MustCompile(`(?i)\b(running|queued|pending|in[ _-]?progress|processing|waiting|started|scheduled|not (yet )?(ready|finished|complete(d)?)|still (working|running|processing))\b`)
var errorRe = regexp.MustCompile(`(?i)(error|fail|denied|unauthori[sz]ed|forbidden|not found|time.?out|unavailable|invalid|exception|rate.?limit|too many requests)|\b(400|401|403|404|408|409|422|429|500|502|503|504)\b`)

func reportsProgress(content string) bool {
	if len(content) > 2000 {
		content = content[:2000]
	}
	return inProgressRe.MatchString(content) && !errorRe.MatchString(content)
}

// history lists the tool calls in order. Identical calls inside one assistant
// message count once, so a model that asks for the same lookup twice in one go is
// not mistaken for a loop. Only the last window calls are kept.
func (l *Layer) history(msgs []core.Message) []entry {
	results := map[string]string{}
	progress := map[string]bool{}
	dry := map[string]bool{}
	for _, m := range msgs {
		if m.Role == core.RoleTool {
			results[m.ToolCallID] = hash(m.Content)
			progress[m.ToolCallID] = l.cfg.ProgressExempt && reportsProgress(m.Content)
			dry[m.ToolCallID] = unproductive(m.Content)
		}
	}
	var out []entry
	for _, m := range msgs {
		if m.Role != core.RoleAssistant || len(m.ToolCalls) == 0 {
			continue
		}
		seen := map[string]bool{}
		for _, c := range m.ToolCalls {
			fp := l.fingerprint(c)
			if seen[fp] {
				continue
			}
			seen[fp] = true
			out = append(out, entry{fp: fp, name: c.Name, result: results[c.ID], progress: progress[c.ID], dry: dry[c.ID]})
		}
	}
	return out
}

// windowed keeps the last Window entries: stuck patterns are about what the agent is doing now.
func (l *Layer) windowed(es []entry) []entry {
	if w := l.cfg.Window; w > 0 && len(es) > w {
		return es[len(es)-w:]
	}
	return es
}

// fingerprint identifies a call by tool and canonical arguments: keys sorted, spacing
// ignored. KeyArgs narrows a tool to the arguments that define "the same call", and
// IgnoreArgs drops volatile ones (timestamps, request IDs) that would make every call look new.
func (l *Layer) fingerprint(c core.ToolCall) string {
	canon := strings.TrimSpace(c.Arguments)
	var args any
	if json.Unmarshal([]byte(c.Arguments), &args) == nil {
		if m, ok := args.(map[string]any); ok {
			if keys, ok := l.cfg.KeyArgs[c.Name]; ok {
				kept := map[string]any{}
				for _, k := range keys {
					if v, present := m[k]; present {
						kept[k] = v
					}
				}
				m = kept
			} else {
				for _, k := range l.cfg.IgnoreArgs {
					delete(m, k)
				}
			}
			args = m
		}
		if b, err := json.Marshal(args); err == nil { // map keys are marshalled in sorted order
			canon = string(b)
		}
	}
	return hash(c.Name + "\x00" + canon)
}

func hash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:8])
}

// analyze runs every detector over the history plus the calls the model is about to
// make (cand, which may be empty).
func (l *Layer) analyze(msgs []core.Message, cand []core.ToolCall) []Finding {
	full := l.history(msgs)
	hist := l.windowed(full)
	all := append([]entry(nil), hist...)
	seen := map[string]bool{}
	for _, c := range cand {
		fp := l.fingerprint(c)
		if !seen[fp] {
			seen[fp] = true
			all = append(all, entry{fp: fp, name: c.Name})
		}
	}
	var f []Finding
	f = append(f, l.noProgress(hist)...)
	f = append(f, l.repeats(all)...)
	f = append(f, l.polls(full, cand)...)
	f = append(f, l.dryStreaks(full, cand)...)
	f = append(f, l.cycles(all)...)
	f = append(f, l.textRepeats(msgs)...)
	return f
}

// allowedSet is the tools exempt from the repeat, no-progress and cycle detectors.
func (l *Layer) allowedSet() map[string]bool {
	m := make(map[string]bool, len(l.cfg.AllowRepeat))
	for _, n := range l.cfg.AllowRepeat {
		m[n] = true
	}
	return m
}

func allInSet(es []entry, set map[string]bool) bool {
	for _, e := range es {
		if !set[e.name] {
			return false
		}
	}
	return true
}

func level(count, warn, stop int) (Level, bool) {
	switch {
	case stop > 0 && count >= stop:
		return Stop, true
	case warn > 0 && count >= warn:
		return Warn, true
	}
	return 0, false
}

// noProgress: the same call returned the same result again and again. This is the
// clearest sign of a stuck agent, because nothing changed between the tries.
func (l *Layer) noProgress(hist []entry) []Finding {
	counts := map[string]int{}
	names := map[string]string{}
	allowed := l.allowedSet()
	for _, e := range hist {
		if e.result == "" || allowed[e.name] || e.progress { // a poll legitimately returns the same result until the job is done
			continue
		}
		k := e.fp + e.result
		counts[k]++
		names[k] = e.name
	}
	var out []Finding
	for _, k := range sortedKeys(counts) {
		if lv, ok := level(counts[k], l.cfg.NoProgressWarn, l.cfg.NoProgressStop); ok {
			out = append(out, Finding{Kind: "no-progress", Level: lv, Count: counts[k], Tool: names[k],
				Detail: fmt.Sprintf("%s was called %d times with the same arguments and returned the same result each time", names[k], counts[k])})
		}
	}
	return out
}

// repeats: the same call many times whatever it returned. The limits are higher than
// for no-progress, because polling a job until it finishes is legitimate.
func (l *Layer) repeats(all []entry) []Finding {
	counts := map[string]int{}
	names := map[string]string{}
	polling := pollingSet(all)
	for _, e := range all {
		counts[e.fp]++
		names[e.fp] = e.name
	}
	allowed := l.allowedSet()
	var out []Finding
	for _, k := range sortedKeys(counts) {
		if allowed[names[k]] {
			continue
		}
		if polling[k] {
			continue // see polls
		}
		if lv, ok := level(counts[k], l.cfg.RepeatWarn, l.cfg.RepeatStop); ok {
			out = append(out, Finding{Kind: "repeat", Level: lv, Count: counts[k], Tool: names[k],
				Detail: fmt.Sprintf("%s was called %d times with the same arguments", names[k], counts[k])})
		}
	}
	return out
}

// pollingSet is the fingerprints of calls whose result reported work in progress.
func pollingSet(es []entry) map[string]bool {
	m := map[string]bool{}
	for _, e := range es {
		if e.progress {
			m[e.fp] = true
		}
	}
	return m
}

// polls: a call that keeps answering "still running". Waiting is legitimate, so the limits
// are far higher than for repeats, and they are counted over the whole conversation, not
// the recent window; a poll that never finishes is still a loop.
func (l *Layer) polls(full []entry, cand []core.ToolCall) []Finding {
	if l.cfg.PollWarn <= 0 && l.cfg.PollStop <= 0 {
		return nil
	}
	polling := pollingSet(full)
	counts := map[string]int{}
	names := map[string]string{}
	for _, e := range full {
		if polling[e.fp] {
			counts[e.fp]++
			names[e.fp] = e.name
		}
	}
	for _, c := range cand {
		if fp := l.fingerprint(c); polling[fp] {
			counts[fp]++
		}
	}
	allowed := l.allowedSet()
	var out []Finding
	for _, k := range sortedKeys(counts) {
		if allowed[names[k]] {
			continue
		}
		if lv, ok := level(counts[k], l.cfg.PollWarn, l.cfg.PollStop); ok {
			out = append(out, Finding{Kind: "poll", Level: lv, Count: counts[k], Tool: names[k],
				Detail: fmt.Sprintf("%s was polled %d times and has not finished", names[k], counts[k])})
		}
	}
	return out
}

// dryStreaks: the last several calls to one tool all came back empty or as errors, whatever
// their arguments. This is the loop the repeat detector cannot see: the agent keeps trying
// new variations ("Atlas plan", "Atlas migration plan", "migration plan Atlas") and every
// one finds nothing. The streak is counted over the whole conversation and a single useful
// result ends it.
func (l *Layer) dryStreaks(full []entry, cand []core.ToolCall) []Finding {
	if l.cfg.DryWarn <= 0 && l.cfg.DryStop <= 0 {
		return nil
	}
	streak := map[string]int{}
	for _, e := range full {
		if e.result == "" {
			continue
		}
		if e.dry {
			streak[e.name]++
		} else {
			streak[e.name] = 0
		}
	}
	allowed := l.allowedSet()
	called := map[string]bool{}
	for _, c := range cand {
		called[c.Name] = true
	}
	var out []Finding
	names := make([]string, 0, len(streak))
	for n := range streak {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		if allowed[n] {
			continue
		}
		count := streak[n]
		if called[n] && count > 0 {
			count++ // the call the model is about to make would extend the streak
		}
		if lv, ok := level(count, l.cfg.DryWarn, l.cfg.DryStop); ok {
			out = append(out, Finding{Kind: "dry", Level: lv, Count: count, Tool: n,
				Detail: fmt.Sprintf("the last %d calls to %s all returned nothing usable (empty results or errors)", count, n)})
		}
	}
	return out
}

// cycles: the last calls repeat a pattern of two or three different calls (A,B,A,B,...).
func (l *Layer) cycles(all []entry) []Finding {
	if l.cfg.CycleWarn <= 0 && l.cfg.CycleStop <= 0 {
		return nil
	}
	var out []Finding
	for p := 2; p <= 3; p++ {
		if len(all) < 2*p {
			continue
		}
		pattern := all[len(all)-p:]
		if distinct(pattern) < 2 {
			continue // a single call repeated is the repeat detector's business
		}
		if allowed := l.allowedSet(); allInSet(pattern, allowed) {
			continue // a cycle made only of tools that are meant to repeat (poll, wait, poll, wait)
		}
		reps := 1
		for end := len(all) - p; end-p >= 0; end -= p {
			if !samePattern(all[end-p:end], pattern) {
				break
			}
			reps++
		}
		if lv, ok := level(reps, l.cfg.CycleWarn, l.cfg.CycleStop); ok {
			var names []string
			for _, e := range pattern {
				names = append(names, e.name)
			}
			out = append(out, Finding{Kind: "cycle", Level: lv, Count: reps, Tool: names[0],
				Detail: fmt.Sprintf("the last calls repeat the cycle %s %d times", strings.Join(names, " -> "), reps)})
			break // the shorter period explains the longer
		}
	}
	return out
}

func distinct(es []entry) int {
	s := map[string]bool{}
	for _, e := range es {
		s[e.fp] = true
	}
	return len(s)
}

func samePattern(a, b []entry) bool {
	for i := range a {
		if a[i].fp != b[i].fp {
			return false
		}
	}
	return true
}

// textRepeats: the assistant's last few messages say nearly the same thing. Similarity
// is the Jaccard overlap of character bigrams, as in DeepEval's stagnation metric.
func (l *Layer) textRepeats(msgs []core.Message) []Finding {
	if l.cfg.TextWarn <= 0 && l.cfg.TextStop <= 0 {
		return nil
	}
	var texts []string
	for _, m := range msgs {
		if m.Role == core.RoleAssistant && len(strings.TrimSpace(m.Content)) >= minTextChars {
			texts = append(texts, m.Content)
		}
	}
	run := 1
	for i := len(texts) - 1; i > 0; i-- {
		if similarity(texts[i], texts[i-1]) < l.cfg.TextSimilarity() {
			break
		}
		run++
	}
	if lv, ok := level(run, l.cfg.TextWarn, l.cfg.TextStop); ok {
		return []Finding{{Kind: "text", Level: lv, Count: run,
			Detail: fmt.Sprintf("the last %d assistant messages say almost the same thing", run)}}
	}
	return nil
}

const minTextChars = 40

// similarity is the Jaccard overlap of the character bigrams of a and b (0 to 1).
func similarity(a, b string) float64 {
	ba, bb := bigrams(a), bigrams(b)
	if len(ba) == 0 || len(bb) == 0 {
		return 0
	}
	inter := 0
	for k := range ba {
		if bb[k] {
			inter++
		}
	}
	return float64(inter) / float64(len(ba)+len(bb)-inter)
}

func bigrams(s string) map[[2]rune]bool {
	r := []rune(strings.ToLower(s))
	if len(r) > 2000 {
		r = r[:2000]
	}
	out := map[[2]rune]bool{}
	for i := 0; i+1 < len(r); i++ {
		out[[2]rune{r[i], r[i+1]}] = true
	}
	return out
}

func sortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// worst returns the most serious finding, or nil.
func worst(fs []Finding) *Finding {
	var w *Finding
	for i := range fs {
		if w == nil || fs[i].Level > w.Level || (fs[i].Level == w.Level && fs[i].Count > w.Count) {
			w = &fs[i]
		}
	}
	return w
}
