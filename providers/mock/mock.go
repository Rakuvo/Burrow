// Package mock is a scripted provider for tests and offline runs.
package mock

import (
	"context"
	"fmt"
	"sync"

	"github.com/rakuvo/burrow/core"
)

// Reply is one scripted answer: either a response or an error.
type Reply struct {
	Response *core.Response
	Err      error
}

// Provider returns its replies in order and records every request it saw.
type Provider struct {
	mu       sync.Mutex
	replies  []Reply
	requests []core.Request
}

func New(replies ...Reply) *Provider { return &Provider{replies: replies} }

// Text builds a plain assistant reply.
func Text(s string) Reply {
	return Reply{Response: &core.Response{
		Message:      core.Message{Role: core.RoleAssistant, Content: s},
		FinishReason: "stop",
		Usage:        core.Usage{InputTokens: 10, OutputTokens: 5},
	}}
}

func (p *Provider) Name() string { return "mock" }

func (p *Provider) Complete(_ context.Context, req core.Request) (*core.Response, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.requests = append(p.requests, req.Clone())
	if len(p.replies) == 0 {
		return nil, fmt.Errorf("mock: no scripted reply left")
	}
	r := p.replies[0]
	p.replies = p.replies[1:]
	return r.Response, r.Err
}

// Requests returns every request the provider has received, oldest first.
func (p *Provider) Requests() []core.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]core.Request(nil), p.requests...)
}
