package proxy

import (
	"github.com/rakuvo/burrow/core"
	"io"
	"net/http"
	"time"

	"github.com/rakuvo/burrow/internal/wire"
)

// responses serves POST /v1/responses: the same engine as chat completions
// behind OpenAI's newer request shape. Stateless function calling only; see
// wire.DecodeResponsesRequest for what is refused.
func (s *Server) responses(w http.ResponseWriter, r *http.Request) {
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
	req, err := wire.DecodeResponsesRequest(body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	resp, err := s.engine().Complete(core.WithAgent(r.Context(), r.Header.Get("X-Rakuvo-Agent")), *req)
	if err != nil {
		if r.Context().Err() != nil {
			return // client went away
		}
		s.writeEngineError(w, err)
		return
	}
	out, err := wire.EncodeResponses(resp, randHex(12), time.Now().Unix())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", "encoding response")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
}
