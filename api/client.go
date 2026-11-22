package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/rakuvo/burrow/core"
)

// Tracer is a core.Tracer that ships the gateway's trace events to the API in batches.
// It never blocks or fails a model call: a full queue drops the oldest event, and a
// failed upload is retried once and then dropped with a log line. Like the JSONL tracer
// it sends metadata only, never prompts or responses.
type Tracer struct {
	url, key string
	client   *http.Client
	log      *slog.Logger

	mu      sync.Mutex
	queue   []Event
	max     int
	dropped int
	wake    chan struct{}
	done    chan struct{}
	stopped chan struct{}
}

// NewTracer starts the background sender. baseURL is the API root, for example
// "https://api.rakuvo.example"; Close flushes what is queued.
func NewTracer(baseURL, apiKey string, log *slog.Logger) *Tracer {
	if log == nil {
		log = slog.Default()
	}
	t := &Tracer{url: strings.TrimRight(baseURL, "/") + "/v1/ingest/events", key: apiKey,
		client: &http.Client{Timeout: 10 * time.Second}, log: log, max: 10000,
		wake: make(chan struct{}, 1), done: make(chan struct{}), stopped: make(chan struct{})}
	go t.loop()
	return t
}

func (t *Tracer) Emit(e core.Event) {
	ev := FromCore(e)
	t.mu.Lock()
	if len(t.queue) >= t.max {
		t.queue = t.queue[1:]
		t.dropped++
	}
	t.queue = append(t.queue, ev)
	n := len(t.queue)
	t.mu.Unlock()
	if n >= 100 {
		select {
		case t.wake <- struct{}{}:
		default:
		}
	}
}

// Dropped is how many events were discarded because the queue was full or an upload failed.
func (t *Tracer) Dropped() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.dropped
}

func (t *Tracer) loop() {
	defer close(t.stopped)
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-tick.C:
		case <-t.wake:
		case <-t.done:
			t.flush()
			return
		}
		t.flush()
	}
}

func (t *Tracer) flush() {
	for {
		t.mu.Lock()
		n := min(len(t.queue), maxBatch)
		batch := append([]Event(nil), t.queue[:n]...)
		t.queue = t.queue[n:]
		t.mu.Unlock()
		if n == 0 {
			return
		}
		if !t.send(batch) && !t.send(batch) {
			t.mu.Lock()
			t.dropped += n
			t.mu.Unlock()
			t.log.Warn("rakuvo api: dropped trace events after a failed upload", "events", n)
			return
		}
	}
}

func (t *Tracer) send(batch []Event) bool {
	body, _ := json.Marshal(map[string]any{"events": batch})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, t.url, bytes.NewReader(body))
	if err != nil {
		return false
	}
	req.Header.Set("Authorization", "Bearer "+t.key)
	req.Header.Set("Content-Type", "application/json")
	resp, err := t.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusAccepted
}

// Close sends what is still queued and stops the sender.
func (t *Tracer) Close() {
	close(t.done)
	<-t.stopped
}

// MultiTracer sends each event to every tracer, so the gateway can keep its local
// JSONL trace and report to the API at the same time.
type MultiTracer []core.Tracer

func (m MultiTracer) Emit(e core.Event) {
	for _, t := range m {
		t.Emit(e)
	}
}
