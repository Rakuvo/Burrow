// Package openai is a Provider for any OpenAI-compatible /chat/completions
// endpoint: OpenAI itself, vLLM, Ollama, Together, Groq and similar.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/internal/wire"
)

const (
	// DefaultBaseURL is OpenAI's API root.
	DefaultBaseURL = "https://api.openai.com/v1"

	// MaxTokensLegacy is the field every compatible server understands.
	MaxTokensLegacy = wire.MaxTokensLegacy
	// MaxTokensCompletion is the field OpenAI's reasoning models require.
	MaxTokensCompletion = wire.MaxTokensCompletion

	maxBodyBytes   = 32 << 20
	defaultTimeout = 5 * time.Minute
)

// Config configures a Provider. Only APIKey is needed for api.openai.com.
type Config struct {
	Name    string // reported by Provider.Name; defaults to "openai"
	BaseURL string // defaults to DefaultBaseURL
	APIKey  string // sent as a Bearer token when non-empty (local servers may not need one)

	// MaxTokensField is the JSON field used for Request.MaxTokens. It defaults to
	// MaxTokensLegacy; set MaxTokensCompletion for OpenAI's newer models.
	MaxTokensField string
	// OmitTemperature drops the temperature parameter, which reasoning models reject.
	OmitTemperature bool

	HTTPClient *http.Client
}

// Provider implements core.Provider.
type Provider struct {
	cfg Config
}

// New returns a Provider, filling in defaults.
func New(cfg Config) *Provider {
	if cfg.Name == "" {
		cfg.Name = "openai"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.MaxTokensField == "" {
		cfg.MaxTokensField = MaxTokensLegacy
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: defaultTimeout}
	}
	return &Provider{cfg: cfg}
}

func (p *Provider) Name() string { return p.cfg.Name }

// APIError is a non-2xx answer from the endpoint. It implements core.StatusError.
type APIError struct {
	Status     int
	Type       string
	Code       string
	Message    string
	RetryAfter time.Duration // from the Retry-After header, if present
}

func (e *APIError) Error() string {
	return fmt.Sprintf("openai: HTTP %d: %s", e.Status, e.Message)
}

func (e *APIError) StatusCode() int                   { return e.Status }
func (e *APIError) RetryAfterDuration() time.Duration { return e.RetryAfter }

// Retryable reports whether trying the same request again might succeed.
func (e *APIError) Retryable() bool {
	return e.Status == http.StatusTooManyRequests || e.Status >= 500
}

func (p *Provider) Complete(ctx context.Context, req core.Request) (*core.Response, error) {
	body, err := wire.EncodeRequest(req, wire.RequestOptions{
		MaxTokensField:  p.cfg.MaxTokensField,
		OmitTemperature: p.cfg.OmitTemperature,
	})
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hr.Header.Set("Content-Type", "application/json")
	if p.cfg.APIKey != "" {
		hr.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}

	res, err := p.cfg.HTTPClient.Do(hr)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, maxBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("openai: reading response: %w", err)
	}
	if res.StatusCode < 200 || res.StatusCode > 299 {
		return nil, parseAPIError(res, raw)
	}
	resp, err := wire.DecodeResponse(raw)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	return resp, nil
}

func parseAPIError(res *http.Response, raw []byte) *APIError {
	e := &APIError{Status: res.StatusCode}
	var w struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &w) == nil && w.Error.Message != "" {
		e.Message, e.Type = w.Error.Message, w.Error.Type
		if w.Error.Code != nil {
			e.Code = fmt.Sprint(w.Error.Code)
		}
	} else {
		e.Message = strings.TrimSpace(string(raw))
		if len(e.Message) > 300 {
			e.Message = e.Message[:300]
		}
		if e.Message == "" {
			e.Message = http.StatusText(res.StatusCode)
		}
	}
	if s, err := strconv.Atoi(res.Header.Get("Retry-After")); err == nil && s >= 0 {
		e.RetryAfter = time.Duration(s) * time.Second
	}
	return e
}
