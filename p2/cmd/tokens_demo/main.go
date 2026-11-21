// Same information, different token counts. Characters are cheap to count
// but tokens are what you pay for. The first run downloads the encoding
// file. Same BPE data as Python tiktoken, via tiktoken-go.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"unicode/utf8"

	"github.com/pkoukk/tiktoken-go"

	"llm-demos/p2/common"
)

func main() {
	enc, err := tiktoken.GetEncoding("o200k_base")
	if err != nil {
		log.Fatal(err)
	}

	// A struct, not a map, so the JSON fields keep their order.
	order := struct {
		Order    int    `json:"order"`
		Customer string `json:"customer"`
		Team     string `json:"team"`
		Urgency  string `json:"urgency"`
		Channel  string `json:"channel"`
		Draft    string `json:"draft"`
	}{4821, "Dana", "shipping", "high", "email", "Replacement ships Tuesday."}
	pretty, _ := json.MarshalIndent(order, "", "  ")
	minified, _ := json.Marshal(order)

	samples := []struct{ name, text string }{
		{"ticket, English (Dana)", common.Tickets["broken_table"]},
		{"ticket, Spanish (Mateo)", common.Tickets["lamp_spanish"]},
		{"ticket, Hindi (same as Dana's)", common.Tickets["table_hindi"]},
		{"system prompt (episode 1)", common.System},
		{"JSON, pretty (indent=2)", string(pretty)},
		{"JSON, minified", string(minified)},
		{"Python code", "def urgency(ticket):\n" +
			"    return \"high\" if \"broken\" in ticket else \"low\""},
		{"a long number", "8675309867530902"},
		{"emoji", "😀🎉👍"},
	}

	fmt.Printf("%-34s %6s %7s %12s\n", "text", "chars", "tokens", "chars/token")
	for _, s := range samples {
		n := len(enc.Encode(s.text, nil, []string{"all"}))
		fmt.Printf("%-34s: %6d %7d %12.1f\n",
			s.name, utf8.RuneCountInString(s.text), n,
			float64(utf8.RuneCountInString(s.text))/float64(n))
	}
}
