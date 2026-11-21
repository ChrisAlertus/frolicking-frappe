// Same question, same model, two temperatures. At 0 the answer is stable;
// at 1.2 it wanders. Run it a few times — the 1.2 line keeps changing.
// Needs a server running (./setup.sh).
package main

import (
	"fmt"
	"log"
	"sort"
	"strings"

	"llm-demos/p2/common"
)

func main() {
	q := "Which team should handle this ticket: shipping, returns, billing or account? " +
		"Answer with one word.\n\n" + common.Tickets["broken_table"]

	for _, temp := range []float64{0, 1.2} {
		answers := map[string]int{}
		for range 10 {
			out, err := common.Chat(
				[]common.Message{{Role: "user", Content: q}},
				map[string]any{"temperature": temp, "max_tokens": 5})
			if err != nil {
				log.Fatal(err)
			}
			normalized := strings.TrimSuffix(
				strings.ToLower(strings.TrimSpace(out.Content())), ".")
			answers[normalized]++
		}
		fmt.Printf("temperature %v: %s\n", temp, pyDict(answers))
	}
}

// pyDict formats a count map the way Python prints a dict, so the output
// matches the other demos: {'returns': 8, 'shipping': 2}
func pyDict(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf("'%s': %d", k, counts[k])
	}
	return "{" + strings.Join(parts, ", ") + "}"
}
