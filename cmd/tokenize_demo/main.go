// What the model actually sees: text -> token IDs. The first run downloads
// the encoding files. Same BPE data as Python tiktoken, via tiktoken-go.
package main

import (
	"fmt"
	"log"
	"strings"
	"unicode/utf8"

	"github.com/pkoukk/tiktoken-go"
)

func main() {
	enc := mustEncoding("o200k_base")  // GPT-4o family encoding
	old := mustEncoding("cl100k_base") // GPT-4 / GPT-3.5 encoding

	prompt := "Plan a 3-day trip to Lisbon for two people on a budget."
	ids := encode(enc, prompt)
	fmt.Println("prompt:", prompt)
	fmt.Println("token ids:", formatIDs(ids))
	fmt.Println("pieces:", pieces(enc, ids))
	fmt.Printf("%d characters -> %d tokens (o200k), %d tokens (cl100k)\n\n",
		utf8.RuneCountInString(prompt), len(ids), len(encode(old, prompt)))

	samples := []struct{ name, text string }{
		{"English", "How are you today?"},
		{"Hindi", "आज आप कैसे हैं?"},
		{"Code", "for i in range(10): print(i)"},
		{"Number", "3.14159265358979"},
		{"Word", "strawberry"},
	}
	for _, s := range samples {
		t := encode(enc, s.text)
		fmt.Printf("%-8s %3d chars -> %2d tokens (o200k), %2d (cl100k)  %s\n",
			s.name,
			utf8.RuneCountInString(s.text),
			len(t),
			len(encode(old, s.text)),
			pieces(enc, t),
		)
	}
}

func mustEncoding(name string) *tiktoken.Tiktoken {
	enc, err := tiktoken.GetEncoding(name)
	if err != nil {
		log.Fatal(err)
	}
	return enc
}

// Encode(text, nil, []string{"all"}) matches tiktoken's Python default:
// no special tokens allowed, and a special token in the text is an error.
func encode(enc *tiktoken.Tiktoken, text string) []int {
	return enc.Encode(text, nil, []string{"all"})
}

func formatIDs(ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = fmt.Sprint(id)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func pieces(enc *tiktoken.Tiktoken, ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = "'" + enc.Decode([]int{id}) + "'"
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
