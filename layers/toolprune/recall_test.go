package toolprune

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/rakuvo/burrow/core"
)

// TestBFCLRecall measures, offline, how often pruning keeps every tool the reference
// answer needs. Set TOOLPRUNE_BFCL_DIR to the bfcl_eval data directory to run it.
func TestBFCLRecall(t *testing.T) {
	dir := os.Getenv("TOOLPRUNE_BFCL_DIR")
	if dir == "" {
		t.Skip("TOOLPRUNE_BFCL_DIR not set")
	}
	cat := os.Getenv("TOOLPRUNE_CATEGORY")
	if cat == "" {
		cat = "multi_turn_long_context"
	}
	docs := map[string][]core.ToolSpec{}
	paths, _ := filepath.Glob(filepath.Join(dir, "multi_turn_func_doc", "*.json"))
	for _, p := range paths {
		f, _ := os.Open(p)
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			var d map[string]any
			if json.Unmarshal(sc.Bytes(), &d) != nil {
				continue
			}
			sch, _ := json.Marshal(d["parameters"])
			docs[strings.TrimSuffix(filepath.Base(p), ".json")] = append(docs[strings.TrimSuffix(filepath.Base(p), ".json")],
				core.ToolSpec{Name: d["name"].(string), Description: d["description"].(string), Schema: string(sch)})
		}
		f.Close()
	}
	truth := map[string][][]string{}
	tf, _ := os.Open(filepath.Join(dir, "possible_answer", "BFCL_v4_"+cat+".json"))
	defer tf.Close()
	sc := bufio.NewScanner(tf)
	sc.Buffer(make([]byte, 1<<22), 1<<22)
	for sc.Scan() {
		var e struct {
			ID    string     `json:"id"`
			Truth [][]string `json:"ground_truth"`
		}
		if json.Unmarshal(sc.Bytes(), &e) == nil {
			truth[e.ID] = e.Truth
		}
	}
	call := regexp.MustCompile(`^\s*(\w+)\(`)
	df, _ := os.Open(filepath.Join(dir, "BFCL_v4_"+cat+".json"))
	defer df.Close()
	sc = bufio.NewScanner(df)
	sc.Buffer(make([]byte, 1<<24), 1<<24)
	for _, keep := range []int{10, 15, 20, 25} {
		l := New(Config{Keep: keep})
		turnsTotal, turnsOK, sentSum, allSum := 0, 0, 0, 0
		df.Seek(0, 0)
		sc = bufio.NewScanner(df)
		sc.Buffer(make([]byte, 1<<24), 1<<24)
		for sc.Scan() {
			var e struct {
				ID       string                `json:"id"`
				Question [][]map[string]string `json:"question"`
				Classes  []string              `json:"involved_classes"`
			}
			if json.Unmarshal(sc.Bytes(), &e) != nil || truth[e.ID] == nil {
				continue
			}
			var tools []core.ToolSpec
			for _, c := range e.Classes {
				tools = append(tools, docs[strings.ToLower(classFile(c))]...)
			}
			if os.Getenv("TOOLPRUNE_ALL_TOOLS") != "" { // a big catalog: every class at once
				tools = nil
				for _, d := range docs {
					tools = append(tools, d...)
				}
			}
			var msgs []core.Message
			for ti, q := range e.Question {
				if len(q) == 0 || ti >= len(truth[e.ID]) {
					continue
				}
				msgs = append(msgs, core.Message{Role: core.RoleUser, Content: q[len(q)-1]["content"]})
				s := &core.Step{Original: core.Request{Tools: tools, Messages: msgs}}
				s.Request = s.Original.Clone()
				l.Before(context.Background(), s)
				have := map[string]bool{}
				for _, x := range s.Request.Tools {
					have[x.Name] = true
				}
				ok := true
				var calls []core.ToolCall
				for _, g := range truth[e.ID][ti] {
					if m := call.FindStringSubmatch(g); m != nil {
						calls = append(calls, core.ToolCall{Name: m[1]})
						if !have[m[1]] {
							ok = false
						}
					}
				}
				turnsTotal++
				if ok {
					turnsOK++
				}
				sentSum += len(s.Request.Tools)
				allSum += len(tools)
				msgs = append(msgs, core.Message{Role: core.RoleAssistant, ToolCalls: calls})
			}
		}
		t.Logf("%s keep=%d: all needed tools present in %d/%d turns (%.1f%%); avg tools sent %.1f of %.1f",
			cat, keep, turnsOK, turnsTotal, 100*float64(turnsOK)/float64(turnsTotal),
			float64(sentSum)/float64(turnsTotal), float64(allSum)/float64(turnsTotal))
	}
}

func classFile(c string) string {
	m := map[string]string{
		"GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api",
		"TwitterAPI": "posting_api", "TicketAPI": "ticket_api", "TradingBot": "trading_bot",
		"TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control", "WebSearchAPI": "web_search",
	}
	if v, ok := m[c]; ok {
		return v
	}
	return c
}
