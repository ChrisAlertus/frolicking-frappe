// Hybrid search over a toy furniture-store corpus: vector similarity
// (nomic task prefixes, unit-norm vectors so dot product = cosine), BM25
// keyword ranking, and the two fused with reciprocal rank fusion.
package main

import (
	"fmt"
	"log"
	"strings"

	"llm-demos/p4/common"
)

func main() {
	common.RequireEmbedServer()
	ids := make([]string, len(common.StoreDocs))
	texts := make([]string, len(common.StoreDocs))
	tokens := make([][]string, len(common.StoreDocs))
	prefixed := make([]string, len(common.StoreDocs))
	for i, doc := range common.StoreDocs {
		ids[i] = doc.ID
		texts[i] = doc.Text
		tokens[i] = common.Tokenize(doc.ID + " " + doc.Text)
		prefixed[i] = "search_document: " + doc.Text
	}
	docs, err := common.Embed(prefixed)
	if err != nil {
		log.Fatal(err)
	}
	bm25 := common.NewBM25(tokens)

	meaning := common.StoreQueries[0].Text
	q, err := common.Embed([]string{"search_query: " + meaning})
	if err != nil {
		log.Fatal(err)
	}
	qv := q.Row(0)
	top := rankDots(docs, qv)

	fmt.Printf("first 6 numbers of its vector: [%s]\n", head6(qv))
	for _, i := range top[:min(3, len(top))] {
		snippet := texts[i]
		if len(snippet) > 60 {
			snippet = snippet[:60]
		}
		fmt.Printf("  cos=%.3f  %s: %s...\n", common.Dot(docs.Row(i), qv), ids[i], snippet)
	}
	warranty := ""
	for _, doc := range common.StoreDocs {
		if doc.ID == "h-warranty-cushion" {
			warranty = doc.Text
		}
	}
	fmt.Printf("  words shared with the warranty chunk: %s\n",
		strings.Join(shared(common.Tokenize(meaning), common.Tokenize(warranty)), ", "))

	fmt.Printf("\n# hybrid comparison (* = expected doc ranked first)\n")
	fmt.Printf("%-44s  %-22s  %-22s  %-22s\n", "question", "vector", "keyword", "hybrid")
	var lines []string
	for _, item := range common.StoreQueries {
		qv, err := common.Embed([]string{"search_query: " + item.Text})
		if err != nil {
			log.Fatal(err)
		}
		vecRank := rankDots(docs, qv.Row(0))
		kwRank := common.RankDesc(bm25.Scores(common.Tokenize(item.Text)))
		hyRank := common.RRF(60, vecRank, kwRank)
		fmt.Printf("%-44s  %-22s  %-22s  %-22s\n",
			item.Text, fmtRank(vecRank, ids, item.Expected),
			fmtRank(kwRank, ids, item.Expected), fmtRank(hyRank, ids, item.Expected))
		if strings.HasPrefix(strings.ToLower(item.Text), "oak-2291") {
			lines = []string{
				"OAK-2291 delivery time",
				"  vector : " + fmtOut(vecRank, ids),
				"  keyword: " + fmtOut(kwRank, ids),
				"  hybrid : " + fmtOut(hyRank, ids),
			}
		}
	}
	if lines != nil {
		fmt.Printf("\n# per-query breakdown\n")
		for i, line := range lines {
			if i == 0 {
				fmt.Printf("%d %s\n", i+1, line)
				continue
			}
			fmt.Println(line)
		}
	}
}

func rankDots(docs common.Matrix, q []float32) []int {
	scores := make([]float64, docs.N)
	for i := 0; i < docs.N; i++ {
		scores[i] = float64(common.Dot(docs.Row(i), q))
	}
	return common.RankDesc(scores)
}

func fmtRank(rank []int, ids []string, expected string) string {
	n := min(3, len(rank))
	top := make([]string, n)
	for i := 0; i < n; i++ {
		top[i] = ids[rank[i]]
	}
	star := " "
	if n > 0 && ids[rank[0]] == expected {
		star = "*"
	}
	return star + strings.Join(top, ",")
}

func fmtOut(rank []int, ids []string) string {
	n := min(3, len(rank))
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		mark := ""
		if i == 0 {
			mark = "*"
		}
		parts[i] = mark + ids[rank[i]]
	}
	return strings.Join(parts, ", ")
}

func head6(v []float32) string {
	n := min(6, len(v))
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		parts[i] = fmt.Sprintf("%.3f", v[i])
	}
	return strings.Join(parts, ", ")
}

func shared(a, b []string) []string {
	set := map[string]struct{}{}
	for _, w := range b {
		set[w] = struct{}{}
	}
	var out []string
	seen := map[string]struct{}{}
	for _, w := range a {
		if _, ok := set[w]; !ok {
			continue
		}
		if _, ok := seen[w]; ok {
			continue
		}
		seen[w] = struct{}{}
		out = append(out, w)
	}
	// Match the Python demo, which prints a sorted set.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
