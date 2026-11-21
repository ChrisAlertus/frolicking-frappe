package common

import (
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestBM25RanksTheMatchingDoc(t *testing.T) {
	corpus := [][]string{
		Tokenize("oak dining table delivery time"),
		Tokenize("lamp shade cord switch"),
		Tokenize("rug wool floor living"),
		Tokenize("sofa cushion warranty pet scratch"),
	}
	scores := NewBM25(corpus).Scores(Tokenize("sofa cushion"))
	rank := RankDesc(scores)
	if rank[0] != 3 || scores[3] <= 0 {
		t.Fatalf("rank = %v, scores = %v", rank, scores)
	}
}

func TestRRFPrefersAgreement(t *testing.T) {
	// 7 is rank 1 in both lists, so it leads the fusion.
	fused := RRF(60, []int{7, 1, 2}, []int{3, 7, 1})
	if fused[0] != 7 {
		t.Fatalf("fused = %v", fused)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	m := RandomUnit(4, 8, 3)
	path := filepath.Join(t.TempDir(), "v.f32emb")
	if err := SaveMatrix(path, m); err != nil {
		t.Fatal(err)
	}
	got, err := LoadMatrix(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.N != m.N || got.Dim != m.Dim || len(got.Data) != len(m.Data) {
		t.Fatalf("shape %+v", got)
	}
	for i := range m.Data {
		if got.Data[i] != m.Data[i] {
			t.Fatalf("data[%d] = %v, want %v", i, got.Data[i], m.Data[i])
		}
	}
}

func TestEmbedNormalizesAndKeepsOrder(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[` +
			`{"index":1,"embedding":[0,3]},` +
			`{"index":0,"embedding":[3,0]}]}`))
	}))
	defer srv.Close()
	prev := EmbedURL
	EmbedURL = srv.URL
	t.Cleanup(func() { EmbedURL = prev })

	m, err := Embed([]string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if m.N != 2 || m.Dim != 2 {
		t.Fatalf("shape %d x %d", m.N, m.Dim)
	}
	if math.Abs(float64(m.Row(0)[0])-1) > 1e-5 || m.Row(0)[1] != 0 {
		t.Fatalf("row0 = %v", m.Row(0))
	}
	if m.Row(1)[0] != 0 || math.Abs(float64(m.Row(1)[1])-1) > 1e-5 {
		t.Fatalf("row1 = %v", m.Row(1))
	}
}

func TestLoadRejectsAPlainFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nope")
	if err := os.WriteFile(path, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadMatrix(path); err == nil {
		t.Fatal("expected an error")
	}
}
