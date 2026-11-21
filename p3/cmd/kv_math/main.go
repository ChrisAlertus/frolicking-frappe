// kv_math: KV cache bytes per token.
// Llama-3.1-8B: 2*32*8*128*2 = 128 KiB/token -> 16 GiB at 128K tokens.
package main

import "fmt"

func main() {
	const layers, kvHeads, headDim = 32, 8, 128    // Llama-3.1-8B
	perToken := 2 * layers * kvHeads * headDim * 2 // K+V, fp16

	fmt.Printf("Llama-3.1-8B KV cache: %d KiB per token\n", perToken/1024)
	for _, tokens := range []int{16_384, 32_768, 131_072} {
		fmt.Printf("  %7s tokens: %5.1f GiB\n",
			comma(tokens), float64(perToken*tokens)/(1<<30))
	}
}

func comma(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
