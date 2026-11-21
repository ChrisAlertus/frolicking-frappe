package ann

import (
	"testing"

	"llm-demos/p4/common"
)

func TestExactFindsItself(t *testing.T) {
	db := common.RandomUnit(30, 16, 1)
	got := Exact(db, db.Slice(0, 5), 1)
	for i, row := range got {
		if len(row) != 1 || row[0] != i {
			t.Fatalf("query %d -> %v", i, row)
		}
	}
}

func TestHNSWFindsItself(t *testing.T) {
	db := common.RandomUnit(80, 16, 2)
	h := NewHNSW(16, 64, 2)
	h.Add(db, 0)
	hits := 0
	for i := 0; i < db.N; i++ {
		ids := h.SearchOne(db.Row(i), 1)
		if len(ids) == 1 && ids[0] == i {
			hits++
		}
	}
	if hits < 72 {
		t.Fatalf("self hits %d / %d", hits, db.N)
	}
}

func TestIVFAgreesWithExactWhenItProbesEveryList(t *testing.T) {
	db := common.RandomUnit(40, 16, 4)
	iv := NewIVF(db, 4)
	q := db.Slice(0, 8)
	want := Exact(db, q, 5)
	got := iv.Search(q, 5, iv.NList())
	rec := common.RecallAtK(got, want, 5)
	if rec < 0.99 {
		t.Fatalf("recall@5 with nprobe=nlist = %.3f", rec)
	}
}
