// Time embedding throughput against the local embedding server, then
// extrapolate: how long a million or a billion chunks would take at the
// measured rate, and the storage for a billion 768-dim float32 vectors
// (1e9 * 768 * 4 bytes = 3.07 TB).
package main

import (
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"llm-demos/p4/common"
)

func main() {
	n := flag.Int("n", 1024, "chunks to embed")
	words := flag.Int("words", 45, "words per chunk")
	flag.Parse()

	common.RequireEmbedServer()
	chunks := makeChunks(*n, *words)
	for i, c := range chunks {
		chunks[i] = "search_document: " + c
	}

	t0 := time.Now()
	if _, err := common.Embed(chunks); err != nil {
		log.Fatal(err)
	}
	elapsed := time.Since(t0).Seconds()
	rate := float64(*n) / elapsed
	hours1M := 1_000_000 / rate / 3600
	hours1B := 1_000_000_000 / rate / 3600
	storageTB := 1_000_000_000 * float64(common.Dim) * 4 / 1e12

	fmt.Printf("embedded %s chunks of ~%d words in %.1f s = %.0f chunks/s on this laptop\n",
		common.Comma(*n), *words, elapsed, rate)
	fmt.Printf("  1 million chunks at this rate: %.1f hours = %.1f days\n", hours1M, hours1M/24)
	fmt.Printf("  1 billion chunks at this rate: %s hours = %.1f days\n", comma1(hours1B), hours1B/24)
	fmt.Printf("  storage for 1 billion vectors of %d float32 numbers: %.2f TB\n", common.Dim, storageTB)
}

var vocab = strings.Fields("warranty sofa cushion fabric velvet delivery table bench oak chair " +
	"lounge indoor outdoor scratch pet cover claim shipping returns window " +
	"white glove packaging stock living dining patio teak performance")

func makeChunks(n, words int) []string {
	out := make([]string, n)
	for i := range out {
		toks := make([]string, words)
		for j := range toks {
			toks[j] = vocab[(i*3+j)%len(vocab)]
		}
		out[i] = strings.Join(toks, " ")
	}
	return out
}

func comma1(v float64) string {
	s := fmt.Sprintf("%.1f", v)
	dot := strings.IndexByte(s, '.')
	if dot < 0 {
		dot = len(s)
	}
	intPart := s[:dot]
	var b strings.Builder
	for i, r := range intPart {
		if i > 0 && (len(intPart)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(r)
	}
	b.WriteString(s[dot:])
	return b.String()
}
