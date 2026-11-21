// Embedding client, tokenization, BM25, and the toy furniture corpus the
// hybrid demo searches.
package common

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
	"unicode"
)

// EmbedURL is the OpenAI-compatible embeddings endpoint. llama-server's
// default for ./p4/setup.sh is http://127.0.0.1:8395/v1.
var EmbedURL = getenv("EMBED_URL", "http://127.0.0.1:8395/v1")

var embedClient = &http.Client{Timeout: 300 * time.Second}

type embResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed posts texts to the embeddings endpoint and returns one unit-length
// row per text. Dot product of two rows is cosine similarity.
func Embed(texts []string) (Matrix, error) {
	raw, err := json.Marshal(map[string]any{"input": texts})
	if err != nil {
		return Matrix{}, err
	}
	res, err := embedClient.Post(EmbedURL+"/embeddings", "application/json", bytes.NewReader(raw))
	if err != nil {
		return Matrix{}, err
	}
	defer res.Body.Close()
	if res.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(res.Body, 200))
		return Matrix{}, fmt.Errorf("%s: %s", res.Status, snippet)
	}
	var out embResponse
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return Matrix{}, err
	}
	if len(out.Data) != len(texts) {
		return Matrix{}, fmt.Errorf("embeddings: sent %d texts, got %d vectors", len(texts), len(out.Data))
	}
	sort.SliceStable(out.Data, func(i, j int) bool { return out.Data[i].Index < out.Data[j].Index })
	dim := len(out.Data[0].Embedding)
	m := Matrix{N: len(texts), Dim: dim, Data: make([]float32, len(texts)*dim)}
	for i, row := range out.Data {
		if len(row.Embedding) != dim {
			return Matrix{}, fmt.Errorf("embeddings: row %d has dim %d, want %d", i, len(row.Embedding), dim)
		}
		copy(m.Row(i), row.Embedding)
		Normalize(m.Row(i))
	}
	return m, nil
}

// RequireEmbedServer exits with a setup hint when llama-server is down.
func RequireEmbedServer() {
	if _, err := Embed([]string{"search_query: ping"}); err != nil {
		fmt.Fprintf(os.Stderr, "Could not reach the embedding server at %s. Start it first:\n"+
			"  ./p4/setup.sh    # or: llama-server -hf nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0 --embedding --pooling mean --port 8395 &\n"+
			"Underlying error: %v\n", EmbedURL, err)
		os.Exit(1)
	}
}

// Tokenize lowercases text and splits on anything that is not a letter or
// digit. "OAK-2291" becomes oak, 2291.
func Tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// RRF fuses ranked id lists with reciprocal rank fusion. Rank 1 (the front
// of a list) contributes 1/(k+1). k is 60 in the hybrid demo.
func RRF(k int, lists ...[]int) []int {
	if k <= 0 {
		k = 60
	}
	score := map[int]float64{}
	for _, list := range lists {
		for rank, id := range list {
			score[id] += 1 / float64(k+rank+1)
		}
	}
	ids := make([]int, 0, len(score))
	for id := range score {
		ids = append(ids, id)
	}
	sort.SliceStable(ids, func(i, j int) bool {
		if score[ids[i]] == score[ids[j]] {
			return ids[i] < ids[j]
		}
		return score[ids[i]] > score[ids[j]]
	})
	return ids
}

// BM25 is Okapi BM25 with the rank_bm25 defaults: k1=1.5, b=0.75, and
// negative idf values replaced by epsilon * mean idf (epsilon=0.25).
type BM25 struct {
	k1, b float64
	docs  []map[string]int
	dl    []int
	avgdl float64
	idf   map[string]float64
}

// NewBM25 indexes one token slice per document.
func NewBM25(corpus [][]string) *BM25 {
	b := &BM25{k1: 1.5, b: 0.75, idf: map[string]float64{}}
	nd := map[string]int{}
	sum := 0
	for _, doc := range corpus {
		freq := make(map[string]int, len(doc))
		for _, w := range doc {
			freq[w]++
		}
		b.docs = append(b.docs, freq)
		b.dl = append(b.dl, len(doc))
		sum += len(doc)
		for w := range freq {
			nd[w]++
		}
	}
	n := float64(len(corpus))
	if n > 0 {
		b.avgdl = float64(sum) / n
	}
	var sumIDF float64
	var neg []string
	for w, freq := range nd {
		idf := math.Log(n-float64(freq)+0.5) - math.Log(float64(freq)+0.5)
		b.idf[w] = idf
		sumIDF += idf
		if idf < 0 {
			neg = append(neg, w)
		}
	}
	if len(b.idf) > 0 {
		eps := 0.25 * sumIDF / float64(len(b.idf))
		for _, w := range neg {
			b.idf[w] = eps
		}
	}
	return b
}

// Scores returns one Okapi score per document. Unknown query terms add
// nothing. A repeated query term is scored again, matching rank_bm25.
func (b *BM25) Scores(query []string) []float64 {
	s := make([]float64, len(b.docs))
	for _, q := range query {
		idf, ok := b.idf[q]
		if !ok {
			continue
		}
		for i, doc := range b.docs {
			f := float64(doc[q])
			denom := f + b.k1*(1-b.b+b.b*float64(b.dl[i])/b.avgdl)
			if denom == 0 {
				continue
			}
			s[i] += idf * (f * (b.k1 + 1) / denom)
		}
	}
	return s
}

// RankDesc returns indexes ordered by descending score. Ties keep the
// lower index first.
func RankDesc(scores []float64) []int {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return scores[idx[i]] > scores[idx[j]] })
	return idx
}

func getenv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

// StoreDoc is one chunk of the toy furniture catalog.
type StoreDoc struct {
	ID, Text string
}

// StoreDocs is the hybrid-search corpus.
var StoreDocs = []StoreDoc{
	{"h-warranty-cushion", "Cushions and fabric carry a 1-year warranty. Pet scratches on sofa cushions may be covered under the fabric warranty if you file a claim."},
	{"p-sof-1180", "SOF-1180 Juno three-seat sofa, performance velvet, soft cushions, living room seating."},
	{"p-out-1502", "OUT-1502 Sol outdoor lounge chair, teak frame, weather-resistant cushions for the patio."},
	{"p-oak-2290", "OAK-2290 solid oak dining bench, natural finish, seats two at the table."},
	{"p-oak-2291", "OAK-2291 solid oak dining table, seats six. Delivery time is 2 to 4 weeks; white-glove delivery is available."},
	{"h-ship-large", "Shipping large furniture: delivery time windows, curb-side drop-off, and scheduling a truck for sofas, tables, and benches."},
	{"h-white-glove", "White-glove delivery: we carry the piece inside, place it, and remove packaging. Available on tables and sofas."},
	{"h-redelivery", "Redelivery policy: if you miss the delivery window we will book a second drop at no extra charge."},
	{"h-returns-window", "Returns window: most items can be returned within 30 days. Tables and benches must be unused and in original packaging."},
	{"p-sof-2201", "SOF-2201 compact loveseat, stain-resistant fabric, in-stock sofa for small rooms."},
	{"h-pet-damage", "Pet damage and accidental scratches: what the sofa warranty does and does not cover when a cat or dog claws the fabric."},
	{"p-chr-0804", "CHR-0804 lounge chair, oak legs, linen upholstery, indoor use only."},
}

// StoreQuery is a question, the kind of match it is, and the doc id that
// should rank first.
type StoreQuery struct {
	Kind, Text, Expected string
}

// StoreQueries are the two questions the hybrid demo prints.
var StoreQueries = []StoreQuery{
	{"meaning", "my cat scratched the sofa cushions, is that covered?", "h-warranty-cushion"},
	{"code", "OAK-2291 delivery time", "p-oak-2291"},
}
