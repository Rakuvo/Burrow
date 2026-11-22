package guard

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"
)

// This is a deliberately small JSON Schema validator. It covers what tool
// schemas use in practice and nothing that reaches outside the document: only
// local "#/..." references resolve, and regular expressions run on Go's RE2
// engine, which has no catastrophic backtracking. Keywords it does not
// understand are reported as unchecked instead of silently guessed at.

const maxSchemaDepth = 32

var (
	intRe = regexp.MustCompile(`^-?(0|[1-9][0-9]*)$`)
	numRe = regexp.MustCompile(`^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$`)

	patterns sync.Map // pattern string -> *regexp.Regexp (nil if it does not compile)

	// unchecked keywords: present in a schema but not enforced here.
	uncheckedKeywords = []string{"not", "if", "patternProperties", "dependentRequired", "dependentSchemas",
		"uniqueItems", "prefixItems", "propertyNames", "contains", "multipleOf", "format"}
)

// checker validates one value against a schema and applies lossless coercions.
type checker struct {
	root        map[string]any
	fixes       []string
	unchecked   map[string]bool
	noCoerce    bool // exact-match pass used to pick anyOf/oneOf branches
	strictExtra bool // unknown properties are errors instead of being dropped
}

// validate checks v against schemaJSON. It returns the (possibly coerced) value,
// the problems found, the repairs applied, and the schema keywords it could not
// enforce. An unparseable schema is reported as unchecked, not as a failure of
// the model.
func validate(schemaJSON string, v any, strictExtra bool) (out any, problems, fixes, unchecked []string) {
	schema, err := parseJSON(schemaJSON)
	root, ok := schema.(map[string]any)
	if err != nil || !ok {
		return v, nil, nil, []string{"schema"}
	}
	c := &checker{root: root, unchecked: map[string]bool{}, strictExtra: strictExtra}
	out, problems = c.check(root, v, "arguments", 0)
	for k := range c.unchecked {
		unchecked = append(unchecked, k)
	}
	sort.Strings(unchecked)
	return out, problems, c.fixes, unchecked
}

func parseJSON(s string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

func (c *checker) fix(path, msg string) { c.fixes = append(c.fixes, path+": "+msg) }

func (c *checker) check(schema, v any, path string, depth int) (any, []string) {
	if depth > maxSchemaDepth {
		c.unchecked["depth"] = true
		return v, nil
	}
	sm, ok := schema.(map[string]any)
	if !ok {
		if b, isBool := schema.(bool); isBool && !b {
			return v, []string{path + ": no value is allowed here"}
		}
		return v, nil
	}
	for _, kw := range uncheckedKeywords {
		if _, has := sm[kw]; has {
			c.unchecked[kw] = true
		}
	}
	if ref, has := sm["$ref"].(string); has {
		target, found := c.resolve(ref)
		if !found {
			c.unchecked["$ref"] = true
			return v, nil
		}
		return c.check(target, v, path, depth+1)
	}

	var errs []string

	if all, has := sm["allOf"].([]any); has {
		for _, sub := range all {
			var e []string
			v, e = c.check(sub, v, path, depth+1)
			errs = append(errs, e...)
		}
	}
	for _, kw := range []string{"anyOf", "oneOf"} {
		alts, has := sm[kw].([]any)
		if !has {
			continue
		}
		var matched bool
		v, matched = c.pickBranch(alts, v, path, depth)
		if !matched {
			errs = append(errs, path+": does not match any allowed alternative")
		}
	}

	if t, has := sm["type"]; has {
		types := stringList(t)
		actual := typeOf(v)
		if len(types) > 0 && !typeMatches(actual, types) {
			nv, how, ok := c.coerce(v, types)
			if !ok {
				return v, append(errs, fmt.Sprintf("%s: expected %s, got %s", path, strings.Join(types, " or "), actual))
			}
			v = nv
			c.fix(path, how)
		}
	}

	if enum, has := sm["enum"].([]any); has {
		var e string
		v, e = c.checkEnum(enum, v, path)
		if e != "" {
			errs = append(errs, e)
		}
	}
	if want, has := sm["const"]; has && !jsonEqual(want, v) {
		errs = append(errs, path+": must equal the required constant value")
	}

	switch val := v.(type) {
	case map[string]any:
		errs = append(errs, c.checkObject(sm, val, path, depth)...)
	case []any:
		var e []string
		v, e = c.checkArray(sm, val, path, depth)
		errs = append(errs, e...)
	case string:
		errs = append(errs, checkString(sm, val, path)...)
	case json.Number:
		errs = append(errs, checkNumber(sm, val, path)...)
	}
	return v, errs
}

// pickBranch returns the first branch that accepts v. It tries exact matches
// before coercions, so a legitimate string is never turned into a number just
// because an integer branch is listed first.
func (c *checker) pickBranch(alts []any, v any, path string, depth int) (any, bool) {
	for _, coerce := range []bool{false, true} {
		for _, alt := range alts {
			trial := &checker{root: c.root, unchecked: c.unchecked, noCoerce: !coerce || c.noCoerce, strictExtra: c.strictExtra}
			nv, errs := trial.check(alt, deepCopy(v), path, depth+1)
			if len(errs) == 0 {
				c.fixes = append(c.fixes, trial.fixes...)
				return nv, true
			}
		}
	}
	return v, false
}

func (c *checker) checkObject(sm map[string]any, obj map[string]any, path string, depth int) []string {
	var errs []string
	props, _ := sm["properties"].(map[string]any)
	required := stringList(sm["required"])
	isRequired := map[string]bool{}
	for _, r := range required {
		isRequired[r] = true
	}
	extra, hasExtra := sm["additionalProperties"]

	for _, k := range sortedKeys(obj) {
		child := path + "." + clean(k, 40)
		sub, known := props[k]
		switch {
		case known:
			// A null for an optional, non-nullable property means "not given".
			if obj[k] == nil && !isRequired[k] && !allowsNull(sub) {
				delete(obj, k)
				c.fix(child, "dropped null for an optional property")
				continue
			}
			nv, e := c.check(sub, obj[k], child, depth+1)
			obj[k] = nv
			errs = append(errs, e...)
		case hasExtra && extra == false:
			if c.strictExtra {
				errs = append(errs, child+": unexpected property")
			} else {
				delete(obj, k)
				c.fix(child, "dropped a property the tool does not accept")
			}
		case hasExtra:
			if _, isSchema := extra.(map[string]any); isSchema {
				nv, e := c.check(extra, obj[k], child, depth+1)
				obj[k] = nv
				errs = append(errs, e...)
			}
		}
	}
	for _, r := range required {
		if _, has := obj[r]; !has {
			errs = append(errs, path+"."+clean(r, 40)+": missing required property")
		}
	}
	return errs
}

func (c *checker) checkArray(sm map[string]any, arr []any, path string, depth int) (any, []string) {
	var errs []string
	if items, has := sm["items"]; has {
		for i := range arr {
			nv, e := c.check(items, arr[i], fmt.Sprintf("%s[%d]", path, i), depth+1)
			arr[i] = nv
			errs = append(errs, e...)
		}
	}
	if n, ok := intKeyword(sm, "minItems"); ok && len(arr) < n {
		errs = append(errs, fmt.Sprintf("%s: needs at least %d items", path, n))
	}
	if n, ok := intKeyword(sm, "maxItems"); ok && len(arr) > n {
		errs = append(errs, fmt.Sprintf("%s: allows at most %d items", path, n))
	}
	return arr, errs
}

func checkString(sm map[string]any, s, path string) []string {
	var errs []string
	n := utf8.RuneCountInString(s)
	if m, ok := intKeyword(sm, "minLength"); ok && n < m {
		errs = append(errs, fmt.Sprintf("%s: shorter than %d characters", path, m))
	}
	if m, ok := intKeyword(sm, "maxLength"); ok && n > m {
		errs = append(errs, fmt.Sprintf("%s: longer than %d characters", path, m))
	}
	if p, ok := sm["pattern"].(string); ok {
		if re := compilePattern(p); re != nil && !re.MatchString(s) {
			errs = append(errs, path+": does not match the required pattern")
		}
	}
	return errs
}

func checkNumber(sm map[string]any, n json.Number, path string) []string {
	f, err := n.Float64()
	if err != nil {
		return []string{path + ": not a representable number"}
	}
	var errs []string
	if m, ok := floatKeyword(sm, "minimum"); ok && f < m {
		errs = append(errs, fmt.Sprintf("%s: below the minimum of %v", path, m))
	}
	if m, ok := floatKeyword(sm, "maximum"); ok && f > m {
		errs = append(errs, fmt.Sprintf("%s: above the maximum of %v", path, m))
	}
	if m, ok := floatKeyword(sm, "exclusiveMinimum"); ok && f <= m {
		errs = append(errs, fmt.Sprintf("%s: must be greater than %v", path, m))
	}
	if m, ok := floatKeyword(sm, "exclusiveMaximum"); ok && f >= m {
		errs = append(errs, fmt.Sprintf("%s: must be less than %v", path, m))
	}
	return errs
}

func (c *checker) checkEnum(enum []any, v any, path string) (any, string) {
	for _, e := range enum {
		if jsonEqual(e, v) {
			return v, ""
		}
	}
	// A string that differs only in case or surrounding space, and matches
	// exactly one allowed value, is unambiguous.
	if s, ok := v.(string); ok && !c.noCoerce {
		var hit string
		hits := 0
		for _, e := range enum {
			if es, ok := e.(string); ok && strings.EqualFold(strings.TrimSpace(s), es) {
				hit, hits = es, hits+1
			}
		}
		if hits == 1 {
			c.fix(path, "matched an allowed value ignoring case")
			return hit, ""
		}
	}
	var names []string
	for i, e := range enum {
		if i == 10 {
			names = append(names, "...")
			break
		}
		b, _ := json.Marshal(e)
		names = append(names, clean(string(b), 40))
	}
	return v, fmt.Sprintf("%s: must be one of %s", path, strings.Join(names, ", "))
}

// coerce converts v to one of types only when the conversion is exact and
// unambiguous: "42" to 42, but never "007" to 7 or "4.5" to an integer.
func (c *checker) coerce(v any, types []string) (any, string, bool) {
	if c.noCoerce {
		return v, "", false
	}
	for _, t := range types {
		switch val := v.(type) {
		case string:
			s := strings.TrimSpace(val)
			switch t {
			case "integer":
				if intRe.MatchString(s) {
					return json.Number(s), "read a string as an integer", true
				}
			case "number":
				if numRe.MatchString(s) {
					return json.Number(s), "read a string as a number", true
				}
			case "boolean":
				switch strings.ToLower(s) {
				case "true":
					return true, "read a string as a boolean", true
				case "false":
					return false, "read a string as a boolean", true
				}
			case "object", "array":
				if len(s) > 0 && (s[0] == '{' || s[0] == '[') {
					if pv, err := parseJSON(s); err == nil && typeOf(pv) == t {
						return pv, "decoded JSON that was sent as a string", true
					}
				}
			}
		case json.Number:
			switch t {
			case "string":
				return val.String(), "wrote a number as a string", true
			case "integer":
				if f, err := val.Float64(); err == nil && f == math.Trunc(f) && math.Abs(f) < 1<<53 {
					return json.Number(strconv.FormatInt(int64(f), 10)), "wrote a whole number as an integer", true
				}
			}
		case bool:
			if t == "string" {
				return strconv.FormatBool(val), "wrote a boolean as a string", true
			}
		}
		if t == "array" && v != nil && typeOf(v) != "array" {
			return []any{v}, "wrapped a single value in a list", true
		}
	}
	return v, "", false
}

func (c *checker) resolve(ref string) (any, bool) {
	if !strings.HasPrefix(ref, "#") {
		return nil, false // never fetch anything outside this document
	}
	var cur any = c.root
	if ref == "#" {
		return cur, true
	}
	if !strings.HasPrefix(ref, "#/") {
		return nil, false
	}
	for _, part := range strings.Split(ref[2:], "/") {
		part = strings.ReplaceAll(strings.ReplaceAll(part, "~1", "/"), "~0", "~")
		m, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		if cur, ok = m[part]; !ok {
			return nil, false
		}
	}
	return cur, true
}

// ---------------------------------------------------------------- helpers

func typeOf(v any) string {
	switch val := v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case string:
		return "string"
	case json.Number:
		if isWhole(val) {
			return "integer"
		}
		return "number"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	}
	return "unknown"
}

func isWhole(n json.Number) bool {
	if _, err := n.Int64(); err == nil {
		return true
	}
	f, err := n.Float64()
	return err == nil && f == math.Trunc(f) && !math.IsInf(f, 0)
}

func typeMatches(actual string, types []string) bool {
	for _, t := range types {
		if t == actual || (t == "number" && actual == "integer") {
			return true
		}
	}
	return false
}

func allowsNull(schema any) bool {
	sm, ok := schema.(map[string]any)
	if !ok {
		return true
	}
	for _, t := range stringList(sm["type"]) {
		if t == "null" {
			return true
		}
	}
	for _, kw := range []string{"anyOf", "oneOf"} {
		if alts, ok := sm[kw].([]any); ok {
			for _, a := range alts {
				if allowsNull(a) {
					return true
				}
			}
		}
	}
	_, hasType := sm["type"]
	_, hasRef := sm["$ref"]
	return !hasType && !hasRef && sm["anyOf"] == nil && sm["oneOf"] == nil
}

func stringList(v any) []string {
	switch t := v.(type) {
	case string:
		return []string{t}
	case []any:
		var out []string
		for _, e := range t {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func intKeyword(sm map[string]any, key string) (int, bool) {
	if n, ok := sm[key].(json.Number); ok {
		if i, err := n.Int64(); err == nil {
			return int(i), true
		}
	}
	return 0, false
}

func floatKeyword(sm map[string]any, key string) (float64, bool) {
	if n, ok := sm[key].(json.Number); ok {
		if f, err := n.Float64(); err == nil {
			return f, true
		}
	}
	return 0, false
}

func compilePattern(p string) *regexp.Regexp {
	if cached, ok := patterns.Load(p); ok {
		return cached.(*regexp.Regexp)
	}
	re, err := regexp.Compile(p)
	if err != nil {
		re = nil
	}
	patterns.Store(p, re)
	return re
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func jsonEqual(a, b any) bool {
	ab, err1 := json.Marshal(a)
	bb, err2 := json.Marshal(b)
	return err1 == nil && err2 == nil && bytes.Equal(ab, bb)
}

func deepCopy(v any) any {
	switch t := v.(type) {
	case map[string]any:
		m := make(map[string]any, len(t))
		for k, e := range t {
			m[k] = deepCopy(e)
		}
		return m
	case []any:
		s := make([]any, len(t))
		for i, e := range t {
			s[i] = deepCopy(e)
		}
		return s
	}
	return v
}

// clean makes model-controlled text safe to put in a message that may be shown
// to a person or fed back to a model: control characters become '?', and long
// text is cut.
func clean(s string, max int) string {
	var b strings.Builder
	n := 0
	for _, r := range s {
		if n == max {
			b.WriteString("...")
			break
		}
		if r < 0x20 || r == 0x7f || r == utf8.RuneError {
			r = '?'
		}
		b.WriteRune(r)
		n++
	}
	return b.String()
}
