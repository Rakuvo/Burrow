package guard

import (
	"encoding/json"
	"testing"
)

// The guard parses text chosen by a model, so it must never panic or hang and
// must only ever produce valid JSON when it claims to have repaired something.
func FuzzRepairJSON(f *testing.F) {
	for _, s := range []string{`{"a":1}`, "```json\n{}\n```", `{"a":[1,],}`, `x{"a":"}"}y`, `{"a":`, ``, `[[[[`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, in string) {
		out, _, ok := repairJSON(in)
		if ok && !json.Valid([]byte(out)) {
			t.Fatalf("repairJSON(%q) claimed success with invalid output %q", in, out)
		}
		_ = nestingDepth(in)
	})
}

const fuzzSchema = `{"type":"object","required":["a"],"additionalProperties":false,"properties":{
	"a":{"anyOf":[{"type":"integer"},{"type":"string","pattern":"^[a-z]+$"}]},
	"b":{"type":"array","items":{"$ref":"#/$defs/n"}},"c":{"enum":["x","y"]}},
	"$defs":{"n":{"type":["integer","null"],"minimum":0}}}`

func FuzzValidate(f *testing.F) {
	for _, s := range []string{`{"a":1}`, `{"a":"x","b":["1",null],"c":"X"}`, `{"a":{"a":{}}}`, `[]`, `"{\"a\":1}"`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, in string) {
		v, err := parseJSON(in)
		if err != nil {
			return
		}
		out, probs, _, _ := validate(fuzzSchema, v, false)
		if len(probs) == 0 {
			// Whatever the validator accepts must re-validate cleanly.
			if _, again, _, _ := validate(fuzzSchema, out, false); len(again) != 0 {
				b, _ := json.Marshal(out)
				t.Fatalf("accepted %q as %s, which then fails: %v", in, b, again)
			}
		}
	})
}
