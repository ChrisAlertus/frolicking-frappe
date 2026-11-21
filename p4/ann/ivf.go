package ann

import (
	"math"
	"math/rand/v2"

	"llm-demos/p4/common"
)

// IVF is an in-process IVF-Flat index. Faiss is not linked: the image is
// built with CGO_ENABLED=0, and libfaiss needs cgo. Centroids come from a
// few rounds of spherical k-means on a sample; search probes the nearest
// lists and finishes with exact inner products inside those lists.
type IVF struct {
	db        common.Matrix
	nlist     int
	centroids []float32
	lists     [][]int
}

// NList matches the Python benches: min(1024, max(4, sqrt(n))), and never
// more lists than rows.
func NList(n int) int {
	if n < 1 {
		return 1
	}
	s := int(math.Sqrt(float64(n)))
	if s < 4 {
		s = 4
	}
	if s > 1024 {
		s = 1024
	}
	if s > n {
		s = n
	}
	return s
}

// NewIVF trains on db and assigns every row to one list. db.Data must stay
// alive; the index does not copy the vectors.
func NewIVF(db common.Matrix, seed int64) *IVF {
	nlist := NList(db.N)
	dim := db.Dim
	rng := rand.New(rand.NewPCG(uint64(seed), 7))
	perm := rng.Perm(db.N)

	iv := &IVF{
		db:        db,
		nlist:     nlist,
		centroids: make([]float32, nlist*dim),
		lists:     make([][]int, nlist),
	}
	for c := 0; c < nlist; c++ {
		copy(iv.cent(c), db.Row(perm[c]))
	}

	sampleN := nlist * 32
	if sampleN > db.N {
		sampleN = db.N
	}
	sample := perm[:sampleN]
	assign := make([]int, sampleN)
	for range 5 {
		for i, row := range sample {
			assign[i] = iv.nearestCent(db.Row(row))
		}
		iv.recenter(db, sample, assign, rng)
	}
	for i := 0; i < db.N; i++ {
		c := iv.nearestCent(db.Row(i))
		iv.lists[c] = append(iv.lists[c], i)
	}
	return iv
}

func (iv *IVF) cent(c int) []float32 {
	d := iv.db.Dim
	return iv.centroids[c*d : (c+1)*d]
}

func (iv *IVF) nearestCent(row []float32) int {
	best, bestS := 0, float32(math.Inf(-1))
	for c := 0; c < iv.nlist; c++ {
		s := common.Dot(row, iv.cent(c))
		if s > bestS {
			best, bestS = c, s
		}
	}
	return best
}

func (iv *IVF) recenter(db common.Matrix, sample, assign []int, rng *rand.Rand) {
	dim := db.Dim
	sum := make([]float32, iv.nlist*dim)
	count := make([]int, iv.nlist)
	for i, c := range assign {
		dst := sum[c*dim : (c+1)*dim]
		row := db.Row(sample[i])
		for d := range row {
			dst[d] += row[d]
		}
		count[c]++
	}
	for c := 0; c < iv.nlist; c++ {
		dst := iv.cent(c)
		if count[c] == 0 {
			copy(dst, db.Row(rng.IntN(db.N)))
			continue
		}
		src := sum[c*dim : (c+1)*dim]
		inv := 1 / float32(count[c])
		for d := range dst {
			dst[d] = src[d] * inv
		}
		common.Normalize(dst)
	}
}

// NList reports how many lists the index built.
func (iv *IVF) NList() int { return iv.nlist }

// Search probes the nprobe nearest lists for each query and returns top-k
// database row indexes.
func (iv *IVF) Search(q common.Matrix, k, nprobe int) [][]int {
	if nprobe > iv.nlist {
		nprobe = iv.nlist
	}
	if nprobe < 1 {
		nprobe = 1
	}
	out := make([][]int, q.N)
	for qi := 0; qi < q.N; qi++ {
		qv := q.Row(qi)
		probes := TopK(iv.nlist, nprobe, func(c int) float32 {
			return common.Dot(qv, iv.cent(c))
		})
		var ids []int
		for _, c := range probes {
			ids = append(ids, iv.lists[c]...)
		}
		local := TopK(len(ids), k, func(i int) float32 {
			return common.Dot(qv, iv.db.Row(ids[i]))
		})
		res := make([]int, len(local))
		for i, j := range local {
			res[i] = ids[j]
		}
		out[qi] = res
	}
	return out
}
