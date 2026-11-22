package contextmgr

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/internal/textmatch"
)

// DefaultFactsPrompt asks for the few values a later step may still need from a tool
// output that is about to be masked.
const DefaultFactsPrompt = `You read the output of a tool call and keep only the facts a later step may still need.

Write one line, at most 60 words. Keep, exactly as written: IDs, names, numbers, dates, file paths, statuses, error messages, and the answer or result the call produced.
Drop everything else. Do not explain, do not add advice, do not guess. If nothing in the output matters later, write: nothing needed.`

const maxFactsInput = 6000 // characters of the tool output the extractor is shown

// FactExtractor condenses one tool output to the facts worth keeping.
type FactExtractor interface {
	Extract(ctx context.Context, tool, args, output string) (facts string, usage core.Usage, err error)
}

// LLMFactExtractor asks a small, cheap model. It is a separate provider call from the
// one being supervised, billed as auxiliary usage.
type LLMFactExtractor struct {
	provider  core.Provider
	model     string
	prompt    string
	maxTokens int
	timeout   time.Duration
}

// NewLLMFactExtractor returns an extractor that calls model on provider. An empty
// prompt uses DefaultFactsPrompt; zero maxTokens means 120 and zero timeout 20s.
func NewLLMFactExtractor(p core.Provider, model, prompt string, maxTokens int, timeout time.Duration) *LLMFactExtractor {
	if strings.TrimSpace(prompt) == "" {
		prompt = DefaultFactsPrompt
	}
	if maxTokens <= 0 {
		maxTokens = 120
	}
	if timeout <= 0 {
		timeout = 20 * time.Second
	}
	return &LLMFactExtractor{provider: p, model: model, prompt: prompt, maxTokens: maxTokens, timeout: timeout}
}

func (e *LLMFactExtractor) Extract(ctx context.Context, tool, args, output string) (string, core.Usage, error) {
	ctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()
	temp := 0.0
	resp, err := e.provider.Complete(ctx, core.Request{
		Model:       e.model,
		Temperature: &temp,
		MaxTokens:   e.maxTokens,
		Messages: []core.Message{
			{Role: core.RoleSystem, Content: e.prompt},
			{Role: core.RoleUser, Content: fmt.Sprintf("tool: %s\narguments: %s\noutput:\n%s", tool, cutHead(args, 300), cutMiddle(output, maxFactsInput))},
		},
	})
	if err != nil {
		return "", core.Usage{}, fmt.Errorf("contextmgr: fact extractor call failed: %w", err)
	}
	return strings.TrimSpace(resp.Message.Content), resp.Usage, nil
}

const (
	maxCachedFacts = 8192
	maxFactChars   = 400 // a longer answer is cut: the note must stay far smaller than the output
	factWorkers    = 8
)

// factsFor returns a facts line for each of the given tool results, keyed by index.
// Results already seen are answered from memory, so the same history always masks to
// the same text and a result is only ever extracted (and billed) once. A failed or
// empty extraction gives no entry and the result gets the plain placeholder.
func (l *Layer) factsFor(ctx context.Context, s *core.Step, msgs []core.Message, idx []int, calls map[string]core.ToolCall) map[int]string {
	out := map[int]string{}
	if l.cfg.Facts == nil || len(idx) == 0 {
		return out
	}
	type job struct {
		i   int
		key string
	}
	var todo []job
	l.mu.Lock()
	for _, i := range idx {
		c := calls[msgs[i].ToolCallID]
		sum := sha256.Sum256([]byte(c.Name + "\x00" + c.Arguments + "\x00" + msgs[i].Content))
		key := hex.EncodeToString(sum[:])
		if f, ok := l.facts[key]; ok {
			if f != "" {
				out[i] = f
			}
			continue
		}
		todo = append(todo, job{i, key})
	}
	l.mu.Unlock()
	if len(todo) == 0 {
		return out
	}

	type result struct {
		facts string
		usage core.Usage
		err   error
	}
	results := make([]result, len(todo))
	var wg sync.WaitGroup
	sem := make(chan struct{}, factWorkers)
	for n, j := range todo {
		wg.Add(1)
		sem <- struct{}{}
		go func(n int, j job) {
			defer wg.Done()
			defer func() { <-sem }()
			c := calls[msgs[j.i].ToolCallID]
			f, u, err := l.cfg.Facts.Extract(ctx, c.Name, c.Arguments, msgs[j.i].Content)
			results[n] = result{f, u, err}
		}(n, j)
	}
	wg.Wait()

	failed := 0
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.facts == nil || len(l.facts) >= maxCachedFacts {
		l.facts = map[string]string{}
	}
	for n, j := range todo {
		r := results[n]
		s.AuxUsage.InputTokens += r.usage.InputTokens // real spend, even when the answer is unusable
		s.AuxUsage.OutputTokens += r.usage.OutputTokens
		if r.err != nil {
			failed++
			continue
		}
		f := strings.Join(strings.Fields(r.facts), " ")
		if strings.EqualFold(strings.TrimRight(f, "."), "nothing needed") {
			f = ""
		}
		f = cutHead(f, maxFactChars)
		l.facts[j.key] = f
		if f != "" {
			out[j.i] = f
		}
	}
	if failed > 0 {
		s.Note(l.Name(), fmt.Sprintf("fact extractor failed for %d result(s); masked without a facts note", failed))
	}
	return out
}

// relevant picks, among candidate tool-result indices, the n that best match the
// latest user message and returns them as a set. A result matches by sharing rare
// words with the request, counting both its output and the arguments of its call.
func relevant(msgs []core.Message, cand []int, calls map[string]core.ToolCall, n int) map[int]bool {
	keep := map[int]bool{}
	if n <= 0 || len(cand) == 0 {
		return keep
	}
	query := ""
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == core.RoleUser {
			query = msgs[i].Content
			break
		}
	}
	q := textmatch.Tokens(query)
	if len(q) == 0 {
		return keep
	}
	docs := make([]map[string]bool, len(cand))
	df := map[string]int{}
	for k, i := range cand {
		c := calls[msgs[i].ToolCallID]
		docs[k] = textmatch.Tokens(c.Name + " " + c.Arguments + " " + cutHead(msgs[i].Content, 20000))
		for w := range docs[k] {
			df[w]++
		}
	}
	type sc struct {
		i int
		v float64
	}
	var all []sc
	for k, i := range cand {
		v := 0.0
		for w := range q {
			if docs[k][w] {
				v += 1 / float64(df[w]) // a word found in few results counts for more
			}
		}
		if v > 0 {
			all = append(all, sc{i, v})
		}
	}
	// highest score first; among equals the more recent result wins
	for a := 1; a < len(all); a++ {
		for b := a; b > 0 && (all[b].v > all[b-1].v || (all[b].v == all[b-1].v && all[b].i > all[b-1].i)); b-- {
			all[b], all[b-1] = all[b-1], all[b]
		}
	}
	for k := 0; k < n && k < len(all); k++ {
		keep[all[k].i] = true
	}
	return keep
}
