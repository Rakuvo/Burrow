package guard

import (
	"encoding/json"
	"strings"
)

// repairJSON turns a model's tool-call arguments into valid JSON using only
// fixes that cannot change what the model meant: it strips a markdown fence,
// drops prose around the object, and removes trailing commas. It never closes
// unbalanced brackets or quotes, because that would invent data.
func repairJSON(s string) (out string, fixes []string, ok bool) {
	t := strings.TrimSpace(s)
	if t == "" {
		return "{}", []string{"empty arguments read as {}"}, true
	}
	if json.Valid([]byte(t)) {
		return t, nil, true
	}

	if stripped, did := stripFence(t); did {
		t = stripped
		fixes = append(fixes, "removed markdown fence")
		if json.Valid([]byte(t)) {
			return t, fixes, true
		}
	}
	if i, j := strings.IndexByte(t, '{'), strings.LastIndexByte(t, '}'); i >= 0 && j > i && (i > 0 || j < len(t)-1) {
		t = t[i : j+1]
		fixes = append(fixes, "removed text around the JSON object")
		if json.Valid([]byte(t)) {
			return t, fixes, true
		}
	}
	if stripped, did := dropTrailingCommas(t); did {
		t = stripped
		fixes = append(fixes, "removed trailing comma")
		if json.Valid([]byte(t)) {
			return t, fixes, true
		}
	}
	return s, nil, false
}

func stripFence(t string) (string, bool) {
	if !strings.HasPrefix(t, "```") {
		return t, false
	}
	nl := strings.IndexByte(t, '\n')
	if nl < 0 {
		return t, false
	}
	body := strings.TrimSpace(t[nl+1:])
	body = strings.TrimSpace(strings.TrimSuffix(body, "```"))
	return body, true
}

// dropTrailingCommas removes a comma that is followed only by whitespace and a
// closing bracket, leaving string contents alone.
func dropTrailingCommas(t string) (string, bool) {
	var b strings.Builder
	inStr, esc, changed := false, false, false
	for i := 0; i < len(t); i++ {
		c := t[i]
		switch {
		case inStr:
			if esc {
				esc = false
			} else if c == '\\' {
				esc = true
			} else if c == '"' {
				inStr = false
			}
		case c == '"':
			inStr = true
		case c == ',':
			j := i + 1
			for j < len(t) && (t[j] == ' ' || t[j] == '\t' || t[j] == '\n' || t[j] == '\r') {
				j++
			}
			if j < len(t) && (t[j] == '}' || t[j] == ']') {
				changed = true
				continue
			}
		}
		b.WriteByte(c)
	}
	return b.String(), changed
}

// nestingDepth returns the deepest bracket nesting in s, ignoring brackets
// inside strings. It runs before parsing so a hostile payload is refused
// before it can cost anything.
func nestingDepth(s string) int {
	depth, max := 0, 0
	inStr, esc := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr {
			if esc {
				esc = false
			} else if c == '\\' {
				esc = true
			} else if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{', '[':
			depth++
			if depth > max {
				max = depth
			}
		case '}', ']':
			depth--
		}
	}
	return max
}
