package ann

import (
	"math/rand"

	"github.com/coder/hnsw"

	"llm-demos/p4/common"
)

// HNSW is a thin wrapper around github.com/coder/hnsw. Vectors must already
// be unit length: the graph ranks by cosine distance, which on unit vectors
// is the same order as inner product. The library has no filtered search
// and its Delete removes the node, so the freshness bench keeps its own
// tombstone mask and does not call Delete.
//
// EfSearch is used both while inserting and while searching. Build with the
// value you want as efConstruction, then call SetEf before queries.
type HNSW struct {
	g *hnsw.Graph[int]
}

// NewHNSW builds an empty graph. m is the neighbor cap (32 in the benches).
// ef is the candidate-list size used until SetEf changes it. seed fixes the
// level generator.
func NewHNSW(m, ef int, seed int64) *HNSW {
	g := hnsw.NewGraph[int]()
	g.M = m
	g.Ml = 0.25
	g.EfSearch = ef
	g.Distance = hnsw.CosineDistance
	g.Rng = rand.New(rand.NewSource(seed))
	return &HNSW{g: g}
}

// Add inserts every row. Keys are id0, id0+1, ... The row slices are stored
// in the graph, so m.Data must stay alive.
func (h *HNSW) Add(m common.Matrix, id0 int) {
	for i := 0; i < m.N; i++ {
		h.g.Add(hnsw.MakeNode(id0+i, m.Row(i)))
	}
}

// SetEf changes the search candidate-list size. It does not rebuild.
func (h *HNSW) SetEf(ef int) { h.g.EfSearch = ef }

// Len is the number of vectors still in the graph.
func (h *HNSW) Len() int { return h.g.Len() }

// SearchOne returns up to k neighbor keys, nearest first.
func (h *HNSW) SearchOne(q []float32, k int) []int {
	if k > h.g.Len() {
		k = h.g.Len()
	}
	if k <= 0 {
		return nil
	}
	nodes := h.g.Search(q, k)
	ids := make([]int, len(nodes))
	for i, n := range nodes {
		ids[i] = n.Key
	}
	return ids
}

// Search runs SearchOne for every query row.
func (h *HNSW) Search(q common.Matrix, k int) [][]int {
	out := make([][]int, q.N)
	for i := 0; i < q.N; i++ {
		out[i] = h.SearchOne(q.Row(i), k)
	}
	return out
}
