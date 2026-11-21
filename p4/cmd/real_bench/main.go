// Benchmark over 768-dim embeddings: exact scan vs HNSW vs IVF-Flat,
// reporting recall@10 against exact ground truth and milliseconds per query.
//
// HNSW is github.com/coder/hnsw (pure Go, so the static image can link it).
// IVF-Flat is the in-process index in package ann. Faiss IVF-PQ, the 96-byte
// codes, and the full-vector re-rank pass need libfaiss and are omitted:
// this module builds with CGO_ENABLED=0.
//
// Default synthesizes 235,476 stand-in vectors. Pass -n for a smaller run,
// or -vectors word_embeddings.f32emb from embed_words for real ones.
package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"

	"llm-demos/p4/ann"
	"llm-demos/p4/common"
)

func main() {
	runtime.GOMAXPROCS(1)
	vectors := flag.String("vectors", "", "matrix from embed_words")
	n := flag.Int("n", 235_476, "synthetic rows if -vectors is omitted")
	queries := flag.Int("queries", 500, "held-out queries")
	seed := flag.Int64("seed", 0, "synthetic-vector seed")
	flag.Parse()

	xb, err := loadOrSynth(*vectors, *n, *seed)
	if err != nil {
		log.Fatal(err)
	}
	nq := *queries
	if capN := max(1, xb.N/20); nq > capN {
		nq = capN
	}
	if nq >= xb.N {
		log.Fatal("need at least one database row after holding out queries")
	}
	db := xb.Slice(0, xb.N-nq)
	xq := xb.Slice(xb.N-nq, xb.N)
	nlist := ann.NList(db.N)
	fmt.Printf("database %d x %d  queries %d  nlist=%d\n", db.N, db.Dim, nq, nlist)

	truth := ann.Exact(db, xq, 10)
	perQuery := func(batchMS float64) float64 { return batchMS / float64(nq) }
	report := func(label string, search func() [][]int) {
		ids := search()
		rec := common.RecallAtK(ids, truth, 10)
		ms := perQuery(common.TimedMS(3, func() { search() }))
		fmt.Printf("  %s: recall@10 %.3f, %.2f ms per query\n", label, rec, ms)
	}

	fmt.Printf("\n# exact, graph, clusters\n")
	report("exact scan", func() [][]int { return ann.Exact(db, xq, 10) })

	h := ann.NewHNSW(32, 128, *seed)
	h.Add(db, 0)
	for _, ef := range []int{16, 128, 256} {
		h.SetEf(ef)
		ef := ef
		report(fmt.Sprintf("HNSW M=32 efSearch=%d", ef), func() [][]int { return h.Search(xq, 10) })
	}

	iv := ann.NewIVF(db, *seed)
	for _, nprobe := range []int{1, 16} {
		nprobe := nprobe
		report(fmt.Sprintf("IVF-Flat nprobe=%d", nprobe), func() [][]int {
			return iv.Search(xq, 10, nprobe)
		})
	}

	fmt.Printf("\n# IVF-PQ section\n")
	fmt.Printf("IVF-PQ (96-byte codes + re-rank of the top 100) needs libfaiss and is omitted.\n")
	fmt.Printf("The static image is built with CGO_ENABLED=0, so libfaiss is not linked.\n")
}

func loadOrSynth(path string, n int, seed int64) (common.Matrix, error) {
	if path == "" {
		return common.RandomUnit(n, common.Dim, seed), nil
	}
	xb, err := common.LoadMatrix(path)
	if err != nil {
		return common.Matrix{}, err
	}
	if xb.Dim != common.Dim {
		return common.Matrix{}, fmt.Errorf("expected (N, %d) embeddings in %s, got %d x %d", common.Dim, path, xb.N, xb.Dim)
	}
	return xb, nil
}
