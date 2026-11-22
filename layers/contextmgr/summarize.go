package contextmgr

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"github.com/rakuvo/burrow/core"
)

// DefaultSummaryPrompt tells the summarizer what to keep. The facts an agent needs
// later are specific values (IDs, names, numbers, file paths, what was tried and
// what failed), which a loose summary tends to lose.
const DefaultSummaryPrompt = `You compress the earlier part of a conversation between a user and an assistant that uses tools. Write a summary the assistant can rely on in place of the original.

Keep, exactly as written: names, IDs, numbers, dates, file paths, values the user gave, values tools returned that later steps may need, the state the task is in, what has been done, what failed and why, and anything the user asked for that is still open.
Drop: greetings, repetition, and raw output the assistant has already acted on.
Be brief and factual. Use short lines. Do not add advice or guesses. Do not address the user.`

const (
	maxTranscriptChars = 24000 // what the summarizer is shown
	maxItemChars       = 600   // per message inside the transcript
)

// Summarizer condenses a transcript of old turns.
type Summarizer interface {
	Summarize(ctx context.Context, transcript string) (text string, usage core.Usage, err error)
}

// LLMSummarizer asks a small, cheap model. It is a separate provider call from the
// one being supervised, billed as auxiliary usage.
type LLMSummarizer struct {
	provider  core.Provider
	model     string
	prompt    string
	maxTokens int
	timeout   time.Duration
}

// NewLLMSummarizer returns a summarizer that calls model on provider. An empty
// prompt uses DefaultSummaryPrompt; zero maxTokens means 400 and zero timeout 30s.
func NewLLMSummarizer(p core.Provider, model, prompt string, maxTokens int, timeout time.Duration) *LLMSummarizer {
	if strings.TrimSpace(prompt) == "" {
		prompt = DefaultSummaryPrompt
	}
	if maxTokens <= 0 {
		maxTokens = 400
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &LLMSummarizer{provider: p, model: model, prompt: prompt, maxTokens: maxTokens, timeout: timeout}
}

func (s *LLMSummarizer) Summarize(ctx context.Context, transcript string) (string, core.Usage, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	temp := 0.0
	resp, err := s.provider.Complete(ctx, core.Request{
		Model:       s.model,
		Temperature: &temp,
		MaxTokens:   s.maxTokens,
		Messages: []core.Message{
			{Role: core.RoleSystem, Content: s.prompt},
			{Role: core.RoleUser, Content: transcript},
		},
	})
	if err != nil {
		return "", core.Usage{}, fmt.Errorf("contextmgr: summarizer call failed: %w", err)
	}
	return strings.TrimSpace(resp.Message.Content), resp.Usage, nil
}

// transcript renders old messages as plain text for the summarizer. Long items are
// cut, and the whole thing is bounded, because the text is model- and user-controlled.
func transcript(msgs []core.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		var line string
		switch {
		case m.Role == core.RoleTool:
			line = "tool result: " + cutMiddle(m.Content, maxItemChars)
		case len(m.ToolCalls) > 0:
			var calls []string
			for _, c := range m.ToolCalls {
				calls = append(calls, fmt.Sprintf("%s(%s)", c.Name, cutHead(c.Arguments, 300)))
			}
			line = "assistant called " + strings.Join(calls, "; ")
			if strings.TrimSpace(m.Content) != "" {
				line = "assistant: " + cutMiddle(m.Content, maxItemChars) + "\n" + line
			}
		default:
			line = string(m.Role) + ": " + cutMiddle(m.Content, maxItemChars)
		}
		b.WriteString(line)
		b.WriteString("\n")
		if b.Len() > maxTranscriptChars {
			b.WriteString("[transcript cut here]\n")
			break
		}
	}
	return b.String()
}

func cutMiddle(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return cutHead(s, max/2) + " [...] " + cutTail(s, max-max/2)
}

const maxCachedSummaries = 128

// summarize returns a summary of gone, or "" if there is no summarizer, nothing to
// summarize, or the call failed. Its cost is added to the step's auxiliary usage.
// Identical transcripts are answered from memory, which keeps the prompt prefix the
// same from one call to the next and saves the second summarizer call.
func (l *Layer) summarize(ctx context.Context, s *core.Step, gone []core.Message) string {
	if l.cfg.Summarizer == nil || len(gone) == 0 {
		return ""
	}
	text := transcript(gone)
	sum := sha256.Sum256([]byte(text))
	key := hex.EncodeToString(sum[:])

	l.mu.Lock()
	cached, ok := l.summaries[key]
	l.mu.Unlock()
	if ok {
		return cached
	}

	out, usage, err := l.cfg.Summarizer.Summarize(ctx, text)
	s.AuxUsage.InputTokens += usage.InputTokens // real spend, even when the summary is unusable
	s.AuxUsage.OutputTokens += usage.OutputTokens
	if err != nil {
		s.Note(l.Name(), "summarizer unavailable; old turns dropped instead ("+err.Error()+")")
		return ""
	}
	if out == "" {
		return ""
	}
	l.mu.Lock()
	if l.summaries == nil || len(l.summaries) >= maxCachedSummaries {
		l.summaries = map[string]string{}
	}
	l.summaries[key] = out
	l.mu.Unlock()
	return out
}
