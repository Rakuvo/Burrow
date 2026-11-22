package verify

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/rakuvo/burrow/core"
)

// DefaultJudgePrompt is the system prompt the judge runs with. This is version 3,
// tuned on a development split of the BFCL single-turn data and frozen before the
// held-out split was scored.
const DefaultJudgePrompt = `You route requests. A user sent a request, and an assistant has a set of tools. Decide whether the assistant can carry out the request by calling one of the tools.

Answer APPLIES if both are true:
1. A listed tool does what the user is asking, not merely something on the same topic.
2. A careful assistant could fill in every parameter the tool needs using only this request, taking values the user stated or plainly implied, and without having to guess at facts the user did not give.

Answer NONE if the request is general knowledge, conversation, or a question the tools cannot answer; if the closest tool does something different from what was asked; or if the tool needs a value that the request does not contain and that cannot be taken from it.

Judge by what each tool is for and needs, not by words it shares with the request. Reply with exactly one word: APPLIES or NONE.`

const (
	maxRequestChars = 2000
	maxToolChars    = 300
	maxTools        = 40
)

// Verdict is the judge's answer plus what it cost.
type Verdict struct {
	Applies bool
	Raw     string // the judge's reply, for debugging
	Usage   core.Usage
}

// Judger decides whether any offered tool applies to a request.
type Judger interface {
	Judge(ctx context.Context, req core.Request) (Verdict, error)
}

// Judge asks a small, cheap model. It is a separate provider call from the one
// being supervised, so it never sees the supervised model's answer.
type Judge struct {
	provider core.Provider
	model    string
	prompt   string
	timeout  time.Duration

	mu    sync.Mutex
	cache map[[32]byte]bool // judged requests: the same request and tools get the same answer, free
}

const maxCached = 2048

// NewJudge returns a Judge that calls model on provider. An empty prompt uses
// DefaultJudgePrompt; a zero timeout means 15 seconds.
func NewJudge(p core.Provider, model, prompt string, timeout time.Duration) *Judge {
	if strings.TrimSpace(prompt) == "" {
		prompt = DefaultJudgePrompt
	}
	if timeout <= 0 {
		timeout = 15 * time.Second
	}
	return &Judge{provider: p, model: model, prompt: prompt, timeout: timeout}
}

// Judge classifies req, using only its user messages and its tool names and
// descriptions. Everything is length-limited, because the text is model- and
// user-controlled.
func (j *Judge) Judge(ctx context.Context, req core.Request) (Verdict, error) {
	ctx, cancel := context.WithTimeout(ctx, j.timeout)
	defer cancel()

	text := describe(req)
	key := sha256.Sum256([]byte(text))
	j.mu.Lock()
	applies, hit := j.cache[key]
	j.mu.Unlock()
	if hit {
		return Verdict{Applies: applies, Raw: "cached"}, nil
	}

	temp := 0.0
	resp, err := j.provider.Complete(ctx, core.Request{
		Model:       j.model,
		Temperature: &temp,
		MaxTokens:   8,
		Messages: []core.Message{
			{Role: core.RoleSystem, Content: j.prompt},
			{Role: core.RoleUser, Content: text},
		},
	})
	if err != nil {
		return Verdict{}, fmt.Errorf("verify: judge call failed: %w", err)
	}
	v := Verdict{Raw: strings.TrimSpace(resp.Message.Content), Usage: resp.Usage}
	switch w := strings.ToUpper(strings.TrimSpace(strings.Trim(v.Raw, ".!\"' \n"))); {
	case strings.HasPrefix(w, "APPLIES"):
		v.Applies = true
	case strings.HasPrefix(w, "NONE"):
	default:
		return v, fmt.Errorf("verify: judge gave an unusable answer %q", cut(v.Raw, 40))
	}
	j.mu.Lock()
	if j.cache == nil || len(j.cache) >= maxCached {
		j.cache = map[[32]byte]bool{}
	}
	j.cache[key] = v.Applies
	j.mu.Unlock()
	return v, nil
}

// describe renders the request and the tools in the form the judge reads.
func describe(req core.Request) string {
	var b strings.Builder
	b.WriteString("User request:\n")
	var user []string
	for _, m := range req.Messages {
		if m.Role == core.RoleUser && strings.TrimSpace(m.Content) != "" {
			user = append(user, m.Content)
		}
	}
	b.WriteString(cut(strings.Join(user, "\n"), maxRequestChars))
	b.WriteString("\n\nAvailable tools:\n")
	for i, t := range req.Tools {
		if i == maxTools {
			fmt.Fprintf(&b, "- ...and %d more\n", len(req.Tools)-maxTools)
			break
		}
		fmt.Fprintf(&b, "- %s: %s\n", cut(t.Name, 80), cut(strings.Join(strings.Fields(t.Description), " "), maxToolChars))
		if params := describeParams(t.Schema); params != "" {
			b.WriteString(params)
		}
	}
	return b.String()
}

const (
	maxParams     = 12
	maxParamChars = 120
)

// describeParams lists what a tool needs: its required parameters with their
// types and descriptions, and the names of the optional ones. A tool that is
// merely on the same topic as a request usually cannot be called because a
// required value is missing, and the judge can only see that if it is told what
// the tool requires. The schema is model- and caller-supplied, so everything is
// bounded and anything unparseable is skipped.
func describeParams(schema string) string {
	if strings.TrimSpace(schema) == "" {
		return ""
	}
	var s struct {
		Required   []string `json:"required"`
		Properties map[string]struct {
			Type        any    `json:"type"`
			Description string `json:"description"`
		} `json:"properties"`
	}
	if json.Unmarshal([]byte(schema), &s) != nil || len(s.Properties) == 0 {
		return ""
	}
	isRequired := map[string]bool{}
	for _, r := range s.Required {
		isRequired[r] = true
	}
	names := make([]string, 0, len(s.Properties))
	for n := range s.Properties {
		names = append(names, n)
	}
	sort.Strings(names)

	var req, opt []string
	for _, n := range names {
		if len(req)+len(opt) >= maxParams {
			break
		}
		p := s.Properties[n]
		if !isRequired[n] {
			opt = append(opt, cut(n, 40))
			continue
		}
		desc := cut(strings.Join(strings.Fields(p.Description), " "), maxParamChars)
		if t, ok := p.Type.(string); ok && t != "" {
			desc = t + ": " + desc
		}
		req = append(req, fmt.Sprintf("%s (%s)", cut(n, 40), desc))
	}
	var b strings.Builder
	if len(req) > 0 {
		fmt.Fprintf(&b, "    needs: %s\n", strings.Join(req, "; "))
	} else {
		b.WriteString("    needs: nothing\n")
	}
	if len(opt) > 0 {
		fmt.Fprintf(&b, "    optional: %s\n", strings.Join(opt, ", "))
	}
	return b.String()
}

func cut(s string, max int) string {
	if utf8.RuneCountInString(s) <= max {
		return s
	}
	r := []rune(s)
	return string(r[:max]) + "..."
}
