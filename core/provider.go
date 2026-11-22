package core

import "context"

// Provider turns a Request into a Response. Adapters for OpenAI-compatible
// endpoints, Anthropic and Gemini implement this.
type Provider interface {
	Name() string
	Complete(ctx context.Context, req Request) (*Response, error)
}
