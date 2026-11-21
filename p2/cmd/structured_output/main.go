// A schema beats asking nicely. The prompt is identical in both halves;
// only B constrains the decoding, so B always parses. Note what the schema
// cannot fix: the drafts still mention refunds the prompt told the model
// not to promise. Needs a server running (./setup.sh).
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"llm-demos/p2/common"
)

const n = 20

var schema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"team":    map[string]any{"enum": []string{"shipping", "returns", "billing", "account"}},
		"urgency": map[string]any{"enum": []string{"low", "normal", "high"}},
		"draft":   map[string]any{"type": "string"},
	},
	"required": []string{"team", "urgency", "draft"},
}

type triage struct {
	Team    string `json:"team"`
	Urgency string `json:"urgency"`
	Draft   string `json:"draft"`
}

func (t triage) valid() bool {
	return oneOf(t.Team, "shipping", "returns", "billing", "account") &&
		oneOf(t.Urgency, "low", "normal", "high")
}

func oneOf(v string, options ...string) bool {
	for _, o := range options {
		if v == o {
			return true
		}
	}
	return false
}

func main() {
	messages := []common.Message{
		{Role: "system", Content: common.System},
		{Role: "user", Content: common.Tickets["angry_refund"]},
	}

	// A: ask nicely
	run(messages, "A: prompt only", false)
	// B: same prompt + a schema the server enforces
	run(messages, "B: prompt + enforced schema", true)
}

func run(messages []common.Message, label string, schemaEnforced bool) {
	params := map[string]any{"temperature": 0.8}
	if schemaEnforced {
		params["response_format"] = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "triage",
				"schema": schema,
			},
		}
	}

	valid, refunds := 0, 0
	badExample := ""
	for range n {
		out, err := common.Chat(messages, params)
		if err != nil {
			log.Fatal(err)
		}
		text := out.Content()
		draft := text
		var reply triage
		if err := json.Unmarshal([]byte(text), &reply); err == nil &&
			(!schemaEnforced || reply.valid()) {
			valid++
			draft = reply.Draft
		} else if badExample == "" {
			badExample = text
		}
		if strings.Contains(strings.ToLower(draft), "refund") {
			refunds++
		}
	}

	fmt.Println(label)
	if n-valid > 0 {
		fmt.Printf("  not JSON: %d/%d\n", n-valid, n)
	}
	fmt.Printf("  valid: %d/%d\n", valid, n)
	fmt.Printf("  replies that mention a refund anyway: %d/%d\n", refunds, n)
	if badExample != "" {
		snippet := strings.ReplaceAll(badExample, "\n", "\\n")
		if len(snippet) > 160 {
			snippet = snippet[:160]
		}
		fmt.Printf("  failed reply (first 160 chars): %s\n", snippet)
	}
}
