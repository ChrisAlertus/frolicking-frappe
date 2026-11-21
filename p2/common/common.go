// Package common holds the shared pieces of the support-triage demos: the
// system prompt, a few customer tickets, and a small chat wrapper for any
// OpenAI-compatible /v1/chat/completions endpoint (local llama-server by
// default).
package common

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

var (
	BaseURL = getenv("BASE_URL", "http://127.0.0.1:8080/v1")
	Model   = getenv("MODEL", "local")
	APIKey  = getenv("API_KEY", "none")
)

const System = "You are a support triage assistant for our online furniture store. " +
	"Teams: shipping, returns, billing, account. " +
	"Reply with three lines: 'team: <one team>, 'urgency: low|normal|high', " +
	"then 'draft:' followed by a short, polite reply in the customer's language. " +
	"Never promise refunds or credits; a human decides those."

var Tickets = map[string]string{
	"broken_table": "My wooden dining table arrived yesterday with a deep crack across " +
		"the top and one leg wobbles so badly. Order #4821. I have photos " +
		"if needed. I'd like a replacement sent as soon as possible, " +
		"please — it is our only table.",
	"lamp_spanish": "Hola buenas, la lámpara de pie que pedí llegó con la pantalla " +
		"rota y no enciende. Pedido #7315. ¿Pueden enviarme un repuesto o " +
		"decirme cómo devolverla? Es para el salón de mamá.",
	"table_hindi": "मेरी लकड़ी की डाइनिंग टेबल कल पहुँची, लेकिन उसके ऊपर एक गहरी दरार " +
		"है और एक पैर बहुत हिलता है। ऑर्डर #4821। मेरे पास फोटो हैं, जरूरत " +
		"हो तो मैं भेज सकता हूँ। कृपया जल्द से जल्द नई टेबल भिजवा दें, यह " +
		"हमारी इकलौती डाइनिंग टेबल है।",
	"angry_refund": "This is the second time my order arrived damaged. I want my money " +
		"back right now — refund me in full today or I am calling my bank " +
		"and disputing the charge. Order #6610.",
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Response struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// Content returns the first choice's message content.
func (r *Response) Content() string {
	if len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].Message.Content
}

var client = &http.Client{Timeout: 120 * time.Second}

// Chat performs one chat completion. Extra params (temperature, max_tokens,
// response_format, ...) go straight into the request body.
func Chat(messages []Message, params map[string]any) (*Response, error) {
	body := map[string]any{"model": Model, "messages": messages}
	for k, v := range params {
		body[k] = v
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost,
		BaseURL+"/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+APIKey)
	req.Header.Set("Content-Type", "application/json")

	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(res.Body, 200))
		return nil, fmt.Errorf("%s: %s", res.Status, snippet)
	}
	var out Response
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func getenv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}
