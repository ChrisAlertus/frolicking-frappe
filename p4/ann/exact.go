package ann

import "llm-demos/p4/common"

// TopK returns the indexes of the k highest scores, best first.
// score is called once per index in 0..n-1.
func TopK(n, k int, score func(int) float32) []int {
	if n <= 0 || k <= 0 {
		return nil
	}
	if k > n {
		k = n
	}
	idx := make([]int, 0, k)
	val := make([]float32, 0, k)
	consider := func(i int, s float32) {
		pos := len(idx)
		idx = append(idx, i)
		val = append(val, s)
		for pos > 0 && val[pos] > val[pos-1] {
			val[pos], val[pos-1] = val[pos-1], val[pos]
			idx[pos], idx[pos-1] = idx[pos-1], idx[pos]
			pos--
		}
	}
	for i := 0; i < n; i++ {
		s := score(i)
		if len(idx) < k {
			consider(i, s)
			continue
		}
		if s <= val[k-1] {
			continue
		}
		idx[k-1] = i
		val[k-1] = s
		for pos := k - 1; pos > 0 && val[pos] > val[pos-1]; pos-- {
			val[pos], val[pos-1] = val[pos-1], val[pos]
			idx[pos], idx[pos-1] = idx[pos-1], idx[pos]
		}
	}
	return idx
}

// Exact returns the top-k row indexes of db for each query, by inner product.
func Exact(db, queries common.Matrix, k int) [][]int {
	out := make([][]int, queries.N)
	for q := 0; q < queries.N; q++ {
		qv := queries.Row(q)
		out[q] = TopK(db.N, k, func(i int) float32 {
			return common.Dot(db.Row(i), qv)
		})
	}
	return out
}
