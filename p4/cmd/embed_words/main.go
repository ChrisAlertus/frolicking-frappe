// Embed a word list with the local embedding server and save the matrix for
// real_bench. Full scale is 235,476 words: 235,476 * 768 * 4 bytes = 0.72 GB.
// Without -words, placeholder tokens are synthesized. The file is a small
// header plus raw float32 rows (see common.SaveMatrix), not a .npy.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"llm-demos/p4/common"
)

func main() {
	wordsFile := flag.String("words", "", "one word per line")
	n := flag.Int("n", 235_476, "how many words (use a small value for a quick demo; 0 with -words means the whole file)")
	batch := flag.Int("batch", 256, "texts per embeddings request")
	out := flag.String("out", "word_embeddings.f32emb", "matrix written by common.SaveMatrix")
	flag.Parse()

	common.RequireEmbedServer()
	words, err := loadWords(*wordsFile, *n)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("embedding %d words in batches of %d -> %s\n", len(words), *batch, *out)

	var parts []common.Matrix
	for i := 0; i < len(words); i += *batch {
		j := min(i+*batch, len(words))
		texts := make([]string, j-i)
		for k, w := range words[i:j] {
			texts[k] = "search_document: " + w
		}
		m, err := common.Embed(texts)
		if err != nil {
			log.Fatal(err)
		}
		parts = append(parts, m)
		fmt.Printf("  %d/%d\n", j, len(words))
	}
	xb := stack(parts)
	if err := common.SaveMatrix(*out, xb); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("saved %d x %d float32 (%.2f GB) to %s\n", xb.N, xb.Dim, float64(xb.N*xb.Dim*4)/1e9, *out)
	fmt.Printf("(full scale: 235,476 x %d = 0.72 GB)\n", common.Dim)
}

func loadWords(path string, n int) ([]string, error) {
	if path == "" {
		if n < 1 {
			return nil, fmt.Errorf("-n must be positive when -words is omitted")
		}
		words := make([]string, n)
		for i := range words {
			words[i] = fmt.Sprintf("word%06d", i)
		}
		return words, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var words []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		w := strings.TrimSpace(sc.Text())
		if w == "" || strings.Contains(w, " ") {
			continue
		}
		words = append(words, w)
		if n > 0 && len(words) == n {
			break
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(words) == 0 {
		return nil, fmt.Errorf("%s has no words", path)
	}
	return words, nil
}

func stack(parts []common.Matrix) common.Matrix {
	if len(parts) == 0 {
		return common.Matrix{}
	}
	dim := parts[0].Dim
	n := 0
	for _, p := range parts {
		n += p.N
	}
	data := make([]float32, n*dim)
	off := 0
	for _, p := range parts {
		copy(data[off:], p.Data)
		off += len(p.Data)
	}
	return common.Matrix{N: n, Dim: dim, Data: data}
}
