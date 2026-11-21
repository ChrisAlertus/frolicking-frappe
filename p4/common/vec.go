// Row-major float32 matrices, the embedding file format, and the small
// helpers the benches share.
package common

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"time"
)

const (
	// Dim is the nomic-embed-text-v1.5 width. Every bench assumes it.
	Dim = 768

	embMagic = "p4emb01\n"
)

// Matrix is a row-major float32 matrix. Data is n*dim elements.
type Matrix struct {
	N, Dim int
	Data   []float32
}

func (m Matrix) Row(i int) []float32 {
	return m.Data[i*m.Dim : (i+1)*m.Dim]
}

// Slice returns rows [i, j). The rows share Data's backing array.
func (m Matrix) Slice(i, j int) Matrix {
	return Matrix{N: j - i, Dim: m.Dim, Data: m.Data[i*m.Dim : j*m.Dim]}
}

// Dot is the inner product. On unit vectors that is cosine similarity.
func Dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Normalize scales v to unit length. A zero vector is left alone.
func Normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

// RandomUnit is a stand-in for stored embeddings: n rows of dim, each
// drawn from a standard normal and scaled to unit length. seed makes the
// draw reproducible.
func RandomUnit(n, dim int, seed int64) Matrix {
	rng := rand.New(rand.NewPCG(uint64(seed), uint64(seed>>1)|1))
	data := make([]float32, n*dim)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	for i := 0; i < n; i++ {
		Normalize(data[i*dim : (i+1)*dim])
	}
	return Matrix{N: n, Dim: dim, Data: data}
}

// RecallAtK is the share of true top-k ids recovered in each predicted
// row, then averaged. Order inside the k does not matter.
func RecallAtK(pred, truth [][]int, k int) float64 {
	if len(pred) == 0 || k <= 0 {
		return 0
	}
	hits := 0
	for i := range pred {
		got := make(map[int]struct{}, k)
		for j := 0; j < k && j < len(pred[i]); j++ {
			got[pred[i][j]] = struct{}{}
		}
		for j := 0; j < k && j < len(truth[i]); j++ {
			if _, ok := got[truth[i][j]]; ok {
				hits++
			}
		}
	}
	return float64(hits) / float64(len(pred)*k)
}

// TimedMS runs fn repeats times and returns the mean wall time in
// milliseconds.
func TimedMS(repeats int, fn func()) float64 {
	if repeats < 1 {
		repeats = 1
	}
	start := time.Now()
	for range repeats {
		fn()
	}
	return time.Since(start).Seconds() * 1000 / float64(repeats)
}

// SaveMatrix writes a little-endian embedding file: 8-byte magic, uint32
// row count, uint32 dimension, then row-major float32 values.
func SaveMatrix(path string, m Matrix) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := io.WriteString(f, embMagic); err != nil {
		return err
	}
	hdr := make([]byte, 8)
	binary.LittleEndian.PutUint32(hdr[0:4], uint32(m.N))
	binary.LittleEndian.PutUint32(hdr[4:8], uint32(m.Dim))
	if _, err := f.Write(hdr); err != nil {
		return err
	}
	buf := make([]byte, 4*len(m.Data))
	for i, v := range m.Data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	_, err = f.Write(buf)
	return err
}

// LoadMatrix reads a file written by SaveMatrix.
func LoadMatrix(path string) (Matrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return Matrix{}, err
	}
	defer f.Close()
	magic := make([]byte, len(embMagic))
	if _, err := io.ReadFull(f, magic); err != nil {
		return Matrix{}, err
	}
	if string(magic) != embMagic {
		return Matrix{}, fmt.Errorf("%s is not an embedding file", path)
	}
	hdr := make([]byte, 8)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return Matrix{}, err
	}
	n := int(binary.LittleEndian.Uint32(hdr[0:4]))
	dim := int(binary.LittleEndian.Uint32(hdr[4:8]))
	if n < 0 || dim < 1 {
		return Matrix{}, fmt.Errorf("%s has shape %d x %d", path, n, dim)
	}
	data := make([]float32, n*dim)
	buf := make([]byte, 4*len(data))
	if _, err := io.ReadFull(f, buf); err != nil {
		return Matrix{}, fmt.Errorf("%s: %w", path, err)
	}
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return Matrix{N: n, Dim: dim, Data: data}, nil
}

// Comma inserts thousands separators into a non-negative integer.
func Comma(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
