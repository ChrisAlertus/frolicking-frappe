// Prefix-cache demo against llama.cpp's llama-server.
//
// Server (single slot, so each request reuses or evicts the one cache):
//
//	llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M \
//	    --port 8393 -c 16384 -np 1 --cache-ram 0
package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

var base = getenv("BASE_URL", "http://127.0.0.1:8393")

//go:embed policy.txt
var filler string

var client = &http.Client{Timeout: 120 * time.Second}

func post(path string, body, out any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	res, err := client.Post(base+path, "application/json", bytes.NewReader(raw))
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(res.Body, 200))
		return fmt.Errorf("%s: %s", res.Status, snippet)
	}
	return json.NewDecoder(res.Body).Decode(out)
}

func tokenize(text string) ([]int, error) {
	var out struct {
		Tokens []int `json:"tokens"`
	}
	err := post("/tokenize", map[string]any{"content": text}, &out)
	return out.Tokens, err
}

type completion struct {
	Timings struct {
		PromptN  int     `json:"prompt_n"`
		CacheN   int     `json:"cache_n"`
		PromptMs float64 `json:"prompt_ms"`
	} `json:"timings"`
	WallMs float64
}

func complete(prompt string, nPredict int) (*completion, error) {
	t := time.Now()
	var out completion
	err := post("/completion", map[string]any{
		"prompt":       prompt,
		"n_predict":    nPredict,
		"temperature":  0,
		"cache_prompt": true,
	}, &out)
	out.WallMs = float64(time.Since(t).Microseconds()) / 1000
	return &out, err
}

// prefixOf returns a prompt of exactly nTokens, made by repeating the
// policy text. Trimming matters: with -c 16384 there are only ~380 tokens
// of headroom for the question and the answer, so an overshooting prefix
// would not fit the window.
func prefixOf(nTokens int) (string, error) {
	text := filler
	for {
		tokens, err := tokenize(text)
		if err != nil {
			return "", err
		}
		if len(tokens) >= nTokens {
			var out struct {
				Content string `json:"content"`
			}
			err := post("/detokenize",
				map[string]any{"tokens": tokens[:nTokens]}, &out)
			return out.Content, err
		}
		text += "\n\n" + filler
	}
}

func show(label string, r *completion) {
	t := r.Timings
	fmt.Printf("%-44s computed %6s tokens, reused %6s, prefill %7.0f ms\n",
		label, comma(t.PromptN), comma(t.CacheN), t.PromptMs)
}

func main() {
	doc, err := prefixOf(16000)
	if err != nil {
		fatal(err)
	}
	qa := "\n\nQuestion: can a customer return an unwrapped mattress?\nAnswer:"
	qb := "\n\nQuestion: who pays the stairs fee?\nAnswer:"

	// empty the slot first
	if _, err := complete("Unrelated warm-up so the server holds nothing from the policy.", 1); err != nil {
		fatal(err)
	}
	steps := []struct {
		label  string
		prompt string
	}{
		{"1. cold: 16K policy + question A", doc + qa},
		{"2. same policy + question B (prefix hit)", doc + qb},
		{"3. one word changed at the START + question B", "Note: " + doc + qb},
		{"4. same as 3 again", "Note: " + doc + qb},
	}
	for _, s := range steps {
		r, err := complete(s.prompt, 32)
		if err != nil {
			fatal(err)
		}
		show(s.label, r)
	}
}

func comma(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

func getenv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
