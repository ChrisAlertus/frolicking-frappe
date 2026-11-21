// Stream one chat request over SSE and measure TTFT and tokens/sec.
//
// Works with any OpenAI-compatible /v1/chat/completions endpoint:
// a local llama-server or vLLM (free), or a hosted API (costs money).
// Usage:
//
//	BASE_URL=http://127.0.0.1:8080/v1 MODEL=local go run ./cmd/stream_ttft "your prompt"
//	BASE_URL=https://api.openai.com/v1 MODEL=gpt-4o-mini go run ./cmd/stream_ttft
//
// API_KEY is read from the environment. It is not passed on the command line.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type usageInfo struct {
	CompletionTokens int `json:"completion_tokens"`
	PromptTokens     int `json:"prompt_tokens"`
}

func main() {
	base := getenv("BASE_URL", "http://127.0.0.1:8080/v1")
	prompt := "Explain what a KV cache is in three sentences."
	if len(os.Args) > 1 {
		prompt = os.Args[1]
	}
	body, err := json.Marshal(map[string]any{
		"model":          getenv("MODEL", "local"),
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"messages":       []map[string]string{{"role": "user", "content": prompt}},
	})
	if err != nil {
		fatal(err)
	}

	req, err := http.NewRequest(http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+getenv("API_KEY", "none"))
	req.Header.Set("Content-Type", "application/json")

	// 120s to receive response headers. A Client.Timeout would cancel the
	// whole stream, including a long generation, at 120s.
	client := &http.Client{Transport: &http.Transport{ResponseHeaderTimeout: 120 * time.Second}}
	res, err := client.Do(req)
	if err != nil {
		fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode >= 300 {
		fatal(fmt.Errorf("%s", res.Status))
	}

	t0 := time.Now()
	var first time.Time
	chunks := 0
	var usage *usageInfo
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue // SSE: skip blanks/comments
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var event struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			fatal(err)
		}
		if len(event.Usage) > 0 && string(event.Usage) != "null" {
			usage = &usageInfo{}
			if err := json.Unmarshal(event.Usage, usage); err != nil {
				fatal(err)
			}
		}
		for _, choice := range event.Choices {
			text := choice.Delta.Content
			if text == "" {
				continue
			}
			if first.IsZero() {
				first = time.Now() // time to first token
			}
			chunks++
			fmt.Print(text)
		}
	}
	if err := sc.Err(); err != nil {
		fatal(err)
	}
	end := time.Now()
	if first.IsZero() {
		fatal(fmt.Errorf("stream ended before any tokens"))
	}

	outTokens := chunks
	if usage != nil {
		outTokens = usage.CompletionTokens
	}
	fmt.Printf("\n\nTTFT: %.0f ms\n", first.Sub(t0).Seconds()*1000)
	fmt.Printf("total: %.2f s, output tokens: %d, text chunks: %d\n",
		end.Sub(t0).Seconds(), outTokens, chunks)
	fmt.Printf("decode speed: %.1f tokens/s (after first token)\n",
		float64(outTokens-1)/end.Sub(first).Seconds())
	if usage != nil {
		fmt.Printf("input tokens: %d\n", usage.PromptTokens)
	}
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
