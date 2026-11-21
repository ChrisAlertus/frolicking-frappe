// The million-vector test is synthetic, so this program does not call the
// embedding server. Exact scan is float32 inner product. HNSW is
// github.com/coder/hnsw. IVF-Flat is the in-process index in package ann.
package main

import (
	"flag"
	"fmt"
	"runtime"

	"llm-demos/p4/ann"
	"llm-demos/p4/common"
)

func main() {
	runtime.GOMAXPROCS(1)
	n := flag.Int("n", 1_000_000, "synthetic vectors (default 1,000,000 ≈ 3.07 GB RAM; -n 20000 for a quick demo)")
	queries := flag.Int("queries", 50, "query rows held out of the same draw")
	seed := flag.Int64("seed", 1, "synthetic-vector seed")
	flag.Parse()

	fmt.Printf("synthetic unit vectors: %s x %d  queries %d\n", common.Comma(*n), common.Dim, *queries)
	all := common.RandomUnit(*n+*queries, common.Dim, *seed)
	xq := all.Slice(0, *queries)
	db := all.Slice(*queries, all.N)
	perQuery := func(batchMS float64) float64 { return batchMS / float64(xq.N) }

	truth := ann.Exact(db, xq, 10)
	msExact := perQuery(common.TimedMS(3, func() { ann.Exact(db, xq, 10) }))
	fmt.Printf("exact scan  search %.2f ms per query  recall@10 1.000\n", msExact)

	h := ann.NewHNSW(32, 128, *seed)
	build := common.TimedMS(1, func() { h.Add(db, 0) })
	h.SetEf(128)
	ids := h.Search(xq, 10)
	ms := perQuery(common.TimedMS(3, func() { h.Search(xq, 10) }))
	fmt.Printf("HNSW M=32 efSearch=128  build %.1f s  search %.2f ms per query  recall@10 %.3f\n",
		build/1000, ms, common.RecallAtK(ids, truth, 10))

	var iv *ann.IVF
	train := common.TimedMS(1, func() { iv = ann.NewIVF(db, *seed) })
	ids = iv.Search(xq, 10, 16)
	ms = perQuery(common.TimedMS(3, func() { iv.Search(xq, 10, 16) }))
	fmt.Printf("IVF-Flat nlist=%d nprobe=16  build %.1f s  search %.2f ms per query  recall@10 %.3f\n",
		iv.NList(), train/1000, ms, common.RecallAtK(ids, truth, 10))
}
