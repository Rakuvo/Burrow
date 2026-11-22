// Package proxy serves the engine as an OpenAI-compatible HTTP endpoint, so any
// agent can use Rakuvo by changing its base_url.
package proxy

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/rakuvo/burrow/core"
	"github.com/rakuvo/burrow/internal/wire"
)

const maxRequestBytes = 32 << 20

// Server answers chat-completions requests using Engine.
type Server struct {
	Engine *core.Engine
	// swapped, when set by SetEngine, takes precedence over Engine. Requests already in
	// flight finish on the engine they started with.
	swapped atomic.Pointer[core.Engine]
	// Tokens are the bearer tokens clients may present. Empty means no
	// authentication, which is only appropriate on a loopback address.
	Tokens []string
	Logger *slog.Logger
}

// SetEngine replaces the engine for requests that start from now on, so a gateway can
// change its layers without a restart or dropping a connection.
func (s *Server) SetEngine(e *core.Engine) { s.swapped.Store(e) }

func (s *Server) engine() *core.Engine {
	if e := s.swapped.Load(); e != nil {
		return e
	}
	return s.Engine
}

// Handler returns the HTTP routes. Both /v1/chat/completions and
// /chat/completions are served because clients differ on whether their
// base_url includes /v1.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok")
	})
	chat := s.auth(http.HandlerFunc(s.chat))
	mux.Handle("/v1/chat/completions", chat)
	mux.Handle("/chat/completions", chat)
	responses := s.auth(http.HandlerFunc(s.responses))
	mux.Handle("/v1/responses", responses)
	mux.Handle("/responses", responses)
	return mux
}

func (s *Server) logger() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
}

func (s *Server) auth(next http.Handler) http.Handler {
	if len(s.Tokens) == 0 {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		ok := false
		for _, t := range s.Tokens {
			// Compare every token so timing does not reveal which one matched.
			if subtle.ConstantTimeCompare([]byte(got), []byte(t)) == 1 {
				ok = true
			}
		}
		if !ok {
			writeError(w, http.StatusUnauthorized, "invalid_api_key", "missing or invalid bearer token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "use POST")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, "request_too_large", "request body too large or unreadable")
		return
	}
	parsed, err := wire.DecodeRequest(body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	resp, err := s.engine().Complete(core.WithAgent(r.Context(), r.Header.Get("X-Rakuvo-Agent")), parsed.Request)
	if err != nil {
		if r.Context().Err() != nil {
			return // client went away; nobody to answer
		}
		s.writeEngineError(w, err)
		return
	}

	id, created := "chatcmpl-"+randHex(12), time.Now().Unix()
	if !parsed.Stream {
		out, err := wire.EncodeResponse(resp, id, created)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "internal_error", "encoding response")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(out)
		return
	}

	chunks, err := wire.StreamChunks(resp, id, created, parsed.IncludeUsage)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", "encoding response")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	for _, c := range chunks {
		fmt.Fprintf(w, "data: %s\n\n", c)
		if flusher != nil {
			flusher.Flush()
		}
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
}

// writeEngineError maps an engine failure to what the client should see. A 4xx
// from upstream about the request itself is passed through; anything that is
// our upstream's fault (bad key, outage) is a 502, because the client's own
// credentials are not the problem.
func (s *Server) writeEngineError(w http.ResponseWriter, err error) {
	var se core.StatusError
	if errors.As(err, &se) {
		switch code := se.StatusCode(); code {
		case http.StatusBadRequest, http.StatusNotFound, http.StatusUnprocessableEntity, http.StatusTooManyRequests:
			if d := se.RetryAfterDuration(); d > 0 {
				w.Header().Set("Retry-After", strconv.Itoa(int((d+time.Second-1)/time.Second)))
			}
			writeError(w, code, "upstream_error", se.Error())
		default:
			s.logger().Error("upstream error", "status", code, "err", se)
			writeError(w, http.StatusBadGateway, "upstream_error", fmt.Sprintf("upstream returned HTTP %d", code))
		}
		return
	}
	s.logger().Error("engine error", "err", err)
	var pe core.PublicError
	if errors.As(err, &pe) {
		writeError(w, http.StatusBadGateway, "engine_error", pe.PublicMessage())
		return
	}
	writeError(w, http.StatusBadGateway, "engine_error", "the request could not be completed; see server logs")
}

func writeError(w http.ResponseWriter, status int, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": code, "code": code},
	})
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
