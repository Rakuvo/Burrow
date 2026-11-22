// Package guard is the Response Guard layer. It stands between a model's tool
// calls and whatever will act on them: every call is checked against the tools
// the request offered and their schemas, repaired where the fix is lossless,
// sent back to the model for another try where it is not, and refused if it
// still fails.
package guard

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/rakuvo/burrow/core"
)

// Failure says what to do with a response that is still invalid once the
// engine's attempts run out.
type Failure int

const (
	// Reject fails the request with an error. This is the default: an invalid
	// tool call is never handed on as if it were fine.
	Reject Failure = iota
	// PassThrough returns the last response untouched and records the problem in
	// the trace. Use it only where something downstream does its own validation.
	PassThrough
)

// Rule is an extra, caller-supplied check on a tool call that has already passed
// schema validation, for policy such as "no paths outside the workspace". The
// error text may reach the model and the client, so keep it free of secrets.
type Rule func(call core.ToolCall) error

// Config tunes the guard. The zero value is safe and strict.
type Config struct {
	// MaxToolCalls caps the calls in one response. Default 32.
	MaxToolCalls int
	// MaxArgBytes caps one call's arguments. Default 256 KiB.
	MaxArgBytes int
	// MaxDepth caps JSON nesting in arguments. Default 32.
	MaxDepth int
	// MaxRetryTokens caps the output limit when a truncated response is retried
	// with a larger one. Default 16384.
	MaxRetryTokens int

	// AllowUnknownTools lets calls to tools the request did not offer through
	// unvalidated. Off by default, which is what stops a model from invoking
	// tools it was never given.
	AllowUnknownTools bool
	// NoRepair turns every automatic fix off: calls are validated as sent.
	NoRepair bool
	// StrictExtra rejects properties a schema forbids instead of dropping them.
	StrictExtra bool

	OnFailure Failure
	Rules     []Rule
}

func (c *Config) defaults() {
	if c.MaxToolCalls <= 0 {
		c.MaxToolCalls = 32
	}
	if c.MaxArgBytes <= 0 {
		c.MaxArgBytes = 256 << 10
	}
	if c.MaxDepth <= 0 {
		c.MaxDepth = 32
	}
	if c.MaxRetryTokens <= 0 {
		c.MaxRetryTokens = 16384
	}
}

// Guard implements core.Layer. It keeps no per-request state, so one Guard can
// serve many requests at once.
type Guard struct{ cfg Config }

func New(cfg Config) *Guard {
	cfg.defaults()
	return &Guard{cfg: cfg}
}

func (g *Guard) Name() string { return "guard" }

const (
	keyFeedback  = "guard.feedback"
	keyMaxTokens = "guard.max_tokens"
)

// Before carries what the last attempt learned into the next one.
func (g *Guard) Before(_ context.Context, s *core.Step) error {
	if fb, _ := s.State[keyFeedback].(string); fb != "" {
		s.Request.Messages = append(s.Request.Messages, core.Message{Role: core.RoleUser, Content: fb})
	}
	if n, _ := s.State[keyMaxTokens].(int); n > 0 {
		s.Request.MaxTokens = n
	}
	return nil
}

func (g *Guard) After(_ context.Context, s *core.Step) error {
	if s.Err != nil || s.Response == nil {
		return nil
	}
	resp := *s.Response // never edit the provider's object in place
	tools := map[string]core.ToolSpec{}
	for _, t := range s.Request.Tools {
		tools[t.Name] = t
	}
	truncated := resp.FinishReason == "length"

	var (
		problems []string
		fixes    []string
		checked  = make([]core.ToolCall, 0, len(resp.Message.ToolCalls))
	)
	if n := len(resp.Message.ToolCalls); n > g.cfg.MaxToolCalls {
		problems = append(problems, fmt.Sprintf("%d tool calls in one response; the limit is %d", n, g.cfg.MaxToolCalls))
	} else {
		for i, call := range resp.Message.ToolCalls {
			fixed, fx, probs := g.checkCall(call, tools, truncated)
			checked = append(checked, fixed)
			fixes = append(fixes, prefix(callLabel(i, fixed.Name, tools), fx)...)
			problems = append(problems, prefix(callLabel(i, fixed.Name, tools), probs)...)
		}
		checked, fixes = uniqueIDs(checked, fixes)
		problems = append(problems, g.checkToolChoice(s.Request, checked)...)
	}

	if len(problems) == 0 {
		for _, f := range fixes {
			s.Note(g.Name(), "repaired "+f)
		}
		if len(fixes) > 0 || len(checked) != len(resp.Message.ToolCalls) {
			resp.Message.ToolCalls = checked
			s.Response = &resp
		}
		return nil
	}

	for _, p := range problems {
		s.Note(g.Name(), "invalid: "+p)
	}
	if s.Attempt < s.MaxAttempts {
		s.Retry = true
		s.State[keyFeedback] = feedback(problems)
		if truncated && s.Request.MaxTokens > 0 {
			s.State[keyMaxTokens] = min(s.Request.MaxTokens*2, max(g.cfg.MaxRetryTokens, s.Request.MaxTokens))
		}
		return nil
	}
	if g.cfg.OnFailure == PassThrough {
		s.Note(g.Name(), "passing an invalid response through (OnFailure=PassThrough)")
		return nil
	}
	return &InvalidToolCallError{Problems: problems}
}

// checkCall validates and, where lossless, repairs one tool call.
func (g *Guard) checkCall(call core.ToolCall, tools map[string]core.ToolSpec, truncated bool) (core.ToolCall, []string, []string) {
	var fixes []string

	spec, known := tools[call.Name]
	if !known && !g.cfg.NoRepair {
		if name, ok := matchName(call.Name, tools); ok {
			fixes = append(fixes, fmt.Sprintf("tool name %s read as %s", quote(call.Name), name))
			call.Name, spec, known = name, tools[name], true
		}
	}
	if !known {
		if g.cfg.AllowUnknownTools {
			return call, fixes, nil
		}
		return call, fixes, []string{fmt.Sprintf("unknown tool %s: only the tools offered in this request may be called", quote(call.Name))}
	}

	if len(call.Arguments) > g.cfg.MaxArgBytes {
		return call, fixes, []string{fmt.Sprintf("arguments are %d bytes; the limit is %d", len(call.Arguments), g.cfg.MaxArgBytes)}
	}
	if nestingDepth(call.Arguments) > g.cfg.MaxDepth {
		return call, fixes, []string{fmt.Sprintf("arguments nest deeper than %d levels", g.cfg.MaxDepth)}
	}

	text := call.Arguments
	if g.cfg.NoRepair {
		if !json.Valid([]byte(text)) {
			return call, fixes, []string{"arguments are not valid JSON"}
		}
	} else if truncated && !json.Valid([]byte(strings.TrimSpace(text))) {
		// Closing a cut-off object would invent whatever the model was about to say.
		return call, fixes, []string{"arguments were cut off by the output limit"}
	} else if repaired, fx, ok := repairJSON(text); ok {
		text, fixes = repaired, append(fixes, fx...)
	} else {
		return call, fixes, []string{"arguments are not valid JSON"}
	}

	value, err := parseJSON(text)
	if err != nil {
		return call, fixes, []string{"arguments are not valid JSON"}
	}
	if _, isObj := value.(map[string]any); !isObj {
		if inner, ok := unwrapString(value); ok && !g.cfg.NoRepair {
			value = inner
			fixes = append(fixes, "decoded arguments that were sent as a JSON string")
		} else {
			return call, fixes, []string{"arguments must be a JSON object"}
		}
	}

	var problems []string
	if spec.Schema != "" {
		var schemaFixes []string
		value, problems, schemaFixes, _ = validate(spec.Schema, value, g.cfg.StrictExtra)
		if g.cfg.NoRepair && len(schemaFixes) > 0 {
			// Re-run exactly: with repairs off, a value that needed a fix is a failure.
			problems = append(problems, "arguments needed repair, which is disabled")
		} else {
			fixes = append(fixes, schemaFixes...)
		}
	}
	if len(problems) > 0 {
		return call, fixes, problems
	}

	if len(fixes) > 0 {
		out, err := json.Marshal(value)
		if err != nil {
			return call, fixes, []string{"arguments could not be re-encoded"}
		}
		call.Arguments = string(out)
	}
	for _, rule := range g.cfg.Rules {
		if err := rule(call); err != nil {
			problems = append(problems, "rejected by policy: "+clean(err.Error(), 200))
		}
	}
	return call, fixes, problems
}

// checkToolChoice enforces tool_choice, which the model is free to ignore.
func (g *Guard) checkToolChoice(req core.Request, calls []core.ToolCall) []string {
	raw, ok := req.Extra["tool_choice"]
	if !ok || len(req.Tools) == 0 {
		return nil
	}
	var mode string
	if json.Unmarshal(raw, &mode) == nil {
		switch mode {
		case "required":
			if len(calls) == 0 {
				return []string{"a tool call was required but the response has none"}
			}
		case "none":
			if len(calls) > 0 {
				return []string{"tool calls were not allowed for this request"}
			}
		}
		return nil
	}
	var forced struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &forced) == nil && forced.Function.Name != "" {
		for _, c := range calls {
			if c.Name == forced.Function.Name {
				return nil
			}
		}
		return []string{fmt.Sprintf("a call to %s was required", quote(forced.Function.Name))}
	}
	return nil
}

// matchName finds the one offered tool a slightly-off name refers to, such as
// "Get_Weather" or "functions.get_weather". Ambiguity means no match.
func matchName(name string, tools map[string]core.ToolSpec) (string, bool) {
	norm := func(s string) string {
		if i := strings.LastIndexByte(s, '.'); i >= 0 {
			s = s[i+1:]
		}
		var b strings.Builder
		for _, r := range strings.ToLower(s) {
			if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
				b.WriteRune(r)
			}
		}
		return b.String()
	}
	want := norm(name)
	if want == "" {
		return "", false
	}
	var hit string
	for n := range tools {
		if norm(n) == want {
			if hit != "" {
				return "", false
			}
			hit = n
		}
	}
	return hit, hit != ""
}

func unwrapString(v any) (any, bool) {
	s, ok := v.(string)
	if !ok {
		return nil, false
	}
	inner, err := parseJSON(s)
	if err != nil {
		return nil, false
	}
	_, isObj := inner.(map[string]any)
	return inner, isObj
}

// uniqueIDs gives every call a non-empty, distinct ID, since tool results are
// matched back to calls by ID.
func uniqueIDs(calls []core.ToolCall, fixes []string) ([]core.ToolCall, []string) {
	seen := map[string]bool{}
	for i := range calls {
		if calls[i].ID == "" || seen[calls[i].ID] {
			b := make([]byte, 8)
			_, _ = rand.Read(b)
			calls[i].ID = "call_" + hex.EncodeToString(b)
			fixes = append(fixes, fmt.Sprintf("call %d: assigned a unique id", i+1))
		}
		seen[calls[i].ID] = true
	}
	return calls, fixes
}

func callLabel(i int, name string, tools map[string]core.ToolSpec) string {
	if _, ok := tools[name]; ok {
		return fmt.Sprintf("call %d (%s)", i+1, name)
	}
	return fmt.Sprintf("call %d", i+1)
}

func prefix(label string, msgs []string) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = label + ": " + m
	}
	return out
}

func quote(s string) string { return `"` + clean(s, 64) + `"` }

// feedback is the corrective message for the retry. It is built only from the
// guard's own findings, which have already been cleaned, never from raw model
// output, so a hostile response cannot smuggle instructions into the next turn.
func feedback(problems []string) string {
	const limit = 8
	if len(problems) > limit {
		problems = append(problems[:limit:limit], fmt.Sprintf("...and %d more", len(problems)-limit))
	}
	return "Your previous response had invalid tool calls:\n- " + strings.Join(problems, "\n- ") +
		"\nRespond again with corrected tool calls."
}

// InvalidToolCallError is returned when a response is still invalid after every
// attempt. Its message is safe to show to the client.
type InvalidToolCallError struct{ Problems []string }

func (e *InvalidToolCallError) Error() string {
	return "guard: invalid tool call: " + strings.Join(e.Problems, "; ")
}

func (e *InvalidToolCallError) PublicMessage() string {
	return "the model returned an invalid tool call that could not be repaired: " + strings.Join(e.Problems, "; ")
}
