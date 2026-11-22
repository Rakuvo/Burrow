package core

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// DefaultMaxAttempts bounds retries when Engine.MaxAttempts is unset.
const DefaultMaxAttempts = 3

// Engine runs requests through its layers and a provider.
type Engine struct {
	Provider    Provider
	Layers      []Layer
	Tracer      Tracer
	MaxAttempts int
}

// Complete runs one logical model call. Each attempt resets the request to the
// caller's original, runs every layer's Before, calls the provider, then runs
// every layer's After in reverse. A layer can ask for another attempt by setting
// Step.Retry, up to MaxAttempts. Hard caps are the engine's job, not the layers'.
func (e *Engine) Complete(ctx context.Context, req Request) (*Response, error) {
	if e.Provider == nil {
		return nil, fmt.Errorf("core: engine has no provider")
	}
	tracer := e.Tracer
	if tracer == nil {
		tracer = nopTracer{}
	}
	max := e.MaxAttempts
	if max <= 0 {
		max = DefaultMaxAttempts
	}

	runID := newRunID()
	s := &Step{Original: req.Clone(), State: map[string]any{}, MaxAttempts: max}
	var total Usage // across every attempt

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		s.Attempt++
		s.Retry, s.Err, s.Response, s.notes = false, nil, nil, nil
		s.AuxUsage = Usage{}
		s.Request = s.Original.Clone()

		for _, l := range e.Layers {
			if err := l.Before(ctx, s); err != nil {
				e.emit(tracer, AgentFrom(ctx), runID, s, 0, false)
				return nil, fmt.Errorf("core: layer %s before: %w", l.Name(), err)
			}
		}

		start := time.Now()
		s.Response, s.Err = e.Provider.Complete(ctx, s.Request)
		latency := time.Since(start)
		if s.Response != nil {
			total.InputTokens += s.Response.Usage.InputTokens
			total.OutputTokens += s.Response.Usage.OutputTokens
			total.CachedTokens += s.Response.Usage.CachedTokens
		}

		for i := len(e.Layers) - 1; i >= 0; i-- {
			l := e.Layers[i]
			if err := l.After(ctx, s); err != nil {
				e.emit(tracer, AgentFrom(ctx), runID, s, latency, false)
				return nil, fmt.Errorf("core: layer %s after: %w", l.Name(), err)
			}
		}

		willRetry := s.Retry && s.Attempt < max
		e.emit(tracer, AgentFrom(ctx), runID, s, latency, willRetry)
		if willRetry {
			continue
		}
		if s.Err != nil {
			return nil, s.Err
		}
		if s.Response == nil {
			return nil, fmt.Errorf("core: provider returned no response")
		}
		// Report what the whole call used, retries included: that is what the
		// caller is billed, and an undercount would flatter any cost comparison.
		out := *s.Response
		out.Usage = total
		return &out, nil
	}
}

func (e *Engine) emit(t Tracer, agent, runID string, s *Step, latency time.Duration, retried bool) {
	ev := Event{
		Agent:    agent,
		RunID:    runID,
		Attempt:  s.Attempt,
		Provider: e.Provider.Name(),
		Model:    s.Request.Model,
		Latency:  latency,
		Retried:  retried,
		Notes:    append([]Note(nil), s.notes...),
		AuxUsage: s.AuxUsage,
	}
	if s.Response != nil {
		ev.Usage = s.Response.Usage
	}
	if s.Err != nil {
		ev.Err = s.Err.Error()
	}
	t.Emit(ev)
}

func newRunID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
