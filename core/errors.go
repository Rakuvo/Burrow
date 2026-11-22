package core

import "time"

// PublicError is implemented by errors whose message is safe to show to the
// end client. The proxy shows PublicMessage for these and a generic message for
// every other engine error, which may contain hostnames or other internals.
type PublicError interface {
	error
	PublicMessage() string
}

// StatusError is implemented by provider errors that carry an HTTP status, so
// the proxy can answer its own client with a faithful status code.
type StatusError interface {
	error
	StatusCode() int
	// RetryAfterDuration is the upstream's Retry-After hint, or zero.
	RetryAfterDuration() time.Duration
}
