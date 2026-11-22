// Package textmatch holds the small text-matching helpers the layers share.
package textmatch

import (
	"strings"
	"unicode"
)

var stop = map[string]bool{
	"the": true, "a": true, "an": true, "of": true, "to": true, "and": true, "or": true, "in": true,
	"on": true, "for": true, "is": true, "it": true, "my": true, "me": true, "i": true, "you": true,
	"this": true, "that": true, "with": true, "from": true, "be": true, "as": true, "at": true,
	"by": true, "are": true, "was": true, "can": true, "please": true, "now": true, "then": true,
	"all": true, "get": true, "set": true, "not": true, "if": true, "any": true, "your": true,
}

// Tokens lowercases s and splits it into words at non-alphanumerics, underscores and
// camelCase boundaries, dropping stop words and one-letter words.
func Tokens(s string) map[string]bool {
	out := map[string]bool{}
	var cur []rune
	flush := func() {
		if len(cur) > 1 {
			w := strings.ToLower(string(cur))
			if !stop[w] {
				out[w] = true
				if len(w) > 3 && strings.HasSuffix(w, "s") { // crude plural fold: "files" matches "file"
					out[strings.TrimSuffix(w, "s")] = true
				}
			}
		}
		cur = cur[:0]
	}
	var prev rune
	for _, r := range s {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			if unicode.IsUpper(r) && unicode.IsLower(prev) {
				flush()
			}
			cur = append(cur, r)
		default:
			flush()
		}
		prev = r
	}
	flush()
	return out
}
