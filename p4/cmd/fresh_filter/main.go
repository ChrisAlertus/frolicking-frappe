// HNSW freshness and filtering on synthetic 768-dim vectors: adding new
// items after the build, tombstone deletes (the graph is not told, so its
// length never shrinks), and a rare ~0.8% "in-stock sofas" filter.
//
// coder/hnsw can Delete a node for real, and it has no accept/IDSelector
// callback, so this bench does not call Delete and it cannot filter inside
// the walk. Top-k-then-filter and an exact scan of the matches are the two
// comparisons that the library can run. Faiss IDSelectorBitmap is omitted
// for the same reason IVF-PQ is: libfaiss needs cgo.
//
// Defaults are the full sizes (200,000 products + 1,000 adds + 1,000
// tombstones). Pass -n/-add/-delete for a quick run.
package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"runtime"
	"strconv"
	"strings"
	"time"

	"llm-demos/p4/ann"
	"llm-demos/p4/common"
)

func main() {
	runtime.GOMAXPROCS(1)
	n := flag.Int("n", 200_000, "base products")
	nAdd := flag.Int("add", 1_000, "new items")
	nDel := flag.Int("delete", 1_000, "tombstones")
	seed := flag.Int64("seed", 2, "synthetic-vector seed")
	flag.Parse()

	if *n < 1 || *nAdd < 0 {
		fmt.Println("-n must be positive and -add must not be negative")
		return
	}
	if *nDel > *n {
		*nDel = *n
	}
	if *nDel < 0 {
		*nDel = 0
	}

	xb := common.RandomUnit(*n+*nAdd, common.Dim, *seed)
	base, extra := xb.Slice(0, *n), xb.Slice(*n, xb.N)

	h := ann.NewHNSW(32, 128, *seed)
	t0 := time.Now()
	h.Add(base, 0)
	fmt.Printf("built HNSW over %s products in %.0f s\n", common.Comma(*n), time.Since(t0).Seconds())

	t0 = time.Now()
	h.Add(extra, *n)
	addS := time.Since(t0).Seconds()
	first := 0
	for i := 0; i < *nAdd; i++ {
		ids := h.SearchOne(extra.Row(i), 1)
		if len(ids) == 1 && ids[0] == *n+i {
			first++
		}
	}
	each := 0.0
	if *nAdd > 0 {
		each = addS / float64(*nAdd) * 1000
	}
	fmt.Printf("[new items] added %s new products in %.0f ms (%.2f ms each); searching for each one right after: %d/%d come back first\n",
		common.Comma(*nAdd), addS*1000, each, first, *nAdd)

	ntotalBefore := h.Len()
	tomb := make([]bool, h.Len())
	for i := *n - *nDel; i < *n; i++ {
		tomb[i] = true
	}
	alive := make([]bool, len(tomb))
	for i := range alive {
		alive[i] = !tomb[i]
	}
	q := common.RandomUnit(1, common.Dim, *seed+99).Row(0)
	ids := h.SearchOne(q, min(10_000, h.Len()))
	returnedDeleted := 0
	for _, id := range ids {
		if id >= 0 && id < len(tomb) && tomb[id] {
			returnedDeleted++
		}
	}
	fmt.Printf("[deletes] marked %s products deleted; search returns %d deleted ids in %s results;\n",
		common.Comma(*nDel), returnedDeleted, common.Comma(len(ids)))
	fmt.Printf("        element count before %s, after %s (a delete is a tombstone; Graph.Delete is not called)\n",
		common.Comma(ntotalBefore), common.Comma(h.Len()))
	if returnedDeleted > 0 {
		fmt.Printf("        (raw HNSW still returns tombstoned ids; filter them in application code)\n")
	}

	nAll := h.Len()
	ok := make([]bool, nAll)
	nOK := max(1, int(float64(nAll)*0.008+0.5))
	rng := rand.New(rand.NewPCG(uint64(*seed+7), 1))
	for _, i := range rng.Perm(nAll)[:nOK] {
		ok[i] = true
	}
	for i := range ok {
		ok[i] = ok[i] && alive[i]
	}
	nOK = 0
	var matches []int
	for i, keep := range ok {
		if keep {
			nOK++
			matches = append(matches, i)
		}
	}
	fmt.Printf("\n# Filters: \"in-stock sofas only\"  %s of %s = %.1f%%\n",
		common.Comma(nOK), common.Comma(nAll), 100*float64(nOK)/float64(nAll))

	if len(matches) > 0 {
		q = xb.Row(matches[0])
	}
	filterAfter := func(k int) int {
		got := h.SearchOne(q, k)
		kept := 0
		for _, id := range got {
			if id >= 0 && id < len(ok) && ok[id] {
				kept++
			}
		}
		if kept > 10 {
			kept = 10
		}
		return kept
	}
	for _, spec := range []struct {
		k     int
		label string
	}{{10, "top 10, then filter"}, {100, "top 100, then filter"}} {
		fmt.Printf("  %s: results of 10 = %d\n", spec.label, filterAfter(spec.k))
	}

	fmt.Printf("  filter inside the walk: coder/hnsw has no IDSelector; skipped\n")

	bestLocal := ann.TopK(len(matches), 10, func(i int) float32 {
		return common.Dot(xb.Row(matches[i]), q)
	})
	best := make([]int, len(bestLocal))
	for i, j := range bestLocal {
		best[i] = matches[j]
	}
	fmt.Printf("  exact scan of the %d matches: %s\n", len(matches), fmtIDs(best))
}

func fmtIDs(ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = strconv.Itoa(id)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
