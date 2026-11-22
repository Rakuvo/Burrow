package api

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// ConfigSync keeps a gateway's layer configuration in step with the control plane.
// It fetches once at start (which must succeed, so a gateway never runs with settings
// nobody chose) and then polls. A new configuration that is invalid, or that Apply
// rejects, is logged and ignored: the gateway keeps its last good one. An API outage
// does the same, so the data path never depends on the control plane being up.
type ConfigSync struct {
	URL      string // API root
	Key      string
	Interval time.Duration // default 30s
	Client   *http.Client
	Log      *slog.Logger
	// Apply installs a configuration; return an error to refuse it.
	Apply func(GatewayConfig) error

	etag string
}

// Fetch asks the API for the configuration. changed is false on a 304.
func (c *ConfigSync) Fetch(ctx context.Context) (cfg GatewayConfig, changed bool, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(c.URL, "/")+"/v1/config", nil)
	if err != nil {
		return cfg, false, err
	}
	req.Header.Set("Authorization", "Bearer "+c.Key)
	if c.etag != "" {
		req.Header.Set("If-None-Match", c.etag)
	}
	client := c.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	resp, err := client.Do(req)
	if err != nil {
		return cfg, false, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusNotModified:
		return cfg, false, nil
	case http.StatusOK:
	case http.StatusUnauthorized:
		return cfg, false, fmt.Errorf("the API rejected the key (401)")
	default:
		return cfg, false, fmt.Errorf("the API answered %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxConfigSize+1))
	if err != nil {
		return cfg, false, err
	}
	cfg, err = ParseGatewayConfig(body)
	if err != nil {
		c.etag = resp.Header.Get("ETag") // do not re-report the same bad document on every poll
		return cfg, false, err
	}
	c.etag = resp.Header.Get("ETag")
	return cfg, true, nil
}

func (c *ConfigSync) logger() *slog.Logger {
	if c.Log != nil {
		return c.Log
	}
	return slog.Default()
}

// Start does the first fetch, applies it, and then polls in the background until ctx ends.
func (c *ConfigSync) Start(ctx context.Context) error {
	cfg, _, err := c.Fetch(ctx)
	if err != nil {
		return err
	}
	if err := c.Apply(cfg); err != nil {
		return fmt.Errorf("the configuration cannot be applied: %w", err)
	}
	every := c.Interval
	if every <= 0 {
		every = 30 * time.Second
	}
	go func() {
		t := time.NewTicker(every)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			}
			cfg, changed, err := c.Fetch(ctx)
			if err != nil {
				c.logger().Warn("layer configuration not updated; keeping the last good one", "err", err)
				continue
			}
			if !changed {
				continue
			}
			if err := c.Apply(cfg); err != nil {
				c.logger().Warn("new layer configuration refused; keeping the last good one", "err", err)
			}
		}
	}()
	return nil
}
