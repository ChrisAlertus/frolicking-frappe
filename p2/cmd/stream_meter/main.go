// One streamed triage call, measured. This is where the 'measured'
// shortIn/out numbers in cost_calc come from. Needs a server running
// (./setup.sh) or any OpenAI-compatible endpoint via BASE_URL/MODEL.
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

	"llm-demos/p2/common"
)

func main() {
	body, err := json.Marshal(map[string]any{
		"model":          common.Model,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"messages": []common.Message{
			{Role: "system", Content: common.System},
			{Role: "user", Content: common.Tickets["broken_table"]},
		},
	})
	if err != nil {
		fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost,
		common.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+common.APIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 120 * time.Second}
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
	var usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	}
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
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
			usage = new(struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			})
			if err := json.Unmarshal(event.Usage, usage); err != nil {
				fatal(err)
			}
		}
		for _, choice := range event.Choices {
			if choice.Delta.Content != "" && first.IsZero() {
				first = time.Now()
			}
		}
	}
	if err := sc.Err(); err != nil {
		fatal(err)
	}
	end := time.Now()
	if usage == nil || first.IsZero() {
		fatal(fmt.Errorf("stream ended before any tokens"))
	}

	fmt.Println("--- meter ---")
	fmt.Println("input tokens:", usage.PromptTokens)
	fmt.Println("output tokens:", usage.CompletionTokens)
	fmt.Printf("time to first token: %.0f ms\n", first.Sub(t0).Seconds()*1000)
	fmt.Printf("total: %.2f s\n", end.Sub(t0).Seconds())
	fmt.Printf("decode speed: %.0f tokens/s\n",
		float64(usage.CompletionTokens-1)/end.Sub(first).Seconds())
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
