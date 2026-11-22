package core

import "context"

type agentKey struct{}

// WithAgent returns a context that names which of the caller's agents is making the request, so traces can be split by agent.
func WithAgent(ctx context.Context, name string) context.Context {
	if name == "" {
		return ctx
	}
	if len(name) > 80 {
		name = name[:80]
	}
	return context.WithValue(ctx, agentKey{}, name)
}

// AgentFrom is the agent name set by WithAgent, or "".
func AgentFrom(ctx context.Context) string {
	n, _ := ctx.Value(agentKey{}).(string)
	return n
}
