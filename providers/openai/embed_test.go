package openai

import (
	"strings"
	"testing"
)

func TestCapLengthsCutsOnlyLongTextsWithoutTouchingTheInput(t *testing.T) {
	long := strings.Repeat("é", maxEmbedChars) // 2 bytes each, so well over the cap
	in := []string{"short", long, "also short"}
	got := capLengths(in)
	if len(got[1]) > maxEmbedChars || !strings.HasPrefix(long, got[1]) {
		t.Fatalf("long text not cut to a prefix: %d bytes", len(got[1]))
	}
	if got[1] != strings.ToValidUTF8(got[1], "") {
		t.Fatal("cut split a character")
	}
	if got[0] != "short" || got[2] != "also short" || len(in[1]) != len(long) {
		t.Fatal("other texts or the caller's slice changed")
	}
	if same := capLengths([]string{"a", "b"}); len(same) != 2 {
		t.Fatal("short input altered")
	}
}
