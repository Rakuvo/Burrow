package presets

import "testing"

func TestEveryPresetBuilds(t *testing.T) {
	for _, n := range Names() {
		o := Options{}
		if n == "custom" {
			o.BaseURL = "http://example.test/v1"
		}
		p, err := New(n, o)
		if err != nil || p.Name() != n {
			t.Errorf("%s: %v", n, err)
		}
	}
}

func TestUnknownAndCustomWithoutURL(t *testing.T) {
	if _, err := New("nope", Options{}); err == nil {
		t.Error("unknown provider must fail")
	}
	if _, err := New("custom", Options{}); err == nil {
		t.Error("custom needs a base URL")
	}
	if p, _ := New("anthropic", Options{}); p.Name() != "anthropic" {
		t.Error("anthropic must use the native adapter")
	}
}
