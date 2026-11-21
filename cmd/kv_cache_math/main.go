// KV cache size per token = 2 (K and V) x layers x kv_heads x head_dim x bytes.
// Configs from the models' published config.json files (Hugging Face).
package main

import "fmt"

func main() {
	models := []struct {
		name                     string
		layers, kvHeads, headDim int
	}{
		{"Llama-3.1-8B", 32, 8, 128},
		{"Llama-3.1-70B", 80, 8, 128},
	}
	const bytesPerElem = 2 // fp16 / bf16

	for _, m := range models {
		perToken := 2 * m.layers * m.kvHeads * m.headDim * bytesPerElem
		fmt.Printf("%s: %.0f KiB per token\n", m.name, float64(perToken)/1024)
		for _, ctx := range []int{1_000, 8_000, 128_000} {
			fmt.Printf("   %7s tokens -> %6.2f GiB for ONE conversation\n",
				comma(ctx), float64(perToken*ctx)/(1024*1024*1024))
		}
	}
}

func comma(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
