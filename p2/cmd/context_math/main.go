// Every turn resends the whole conversation, so the input of each turn
// grows in a straight line and the running total grows like a square.
package main

import "fmt"

const (
	first  = 132 // first-turn input tokens
	growth = 260 // extra tokens per turn
)

func turnInput(n int) int { return first + (n-1)*growth }

func total(n int) int {
	sum := 0
	for i := 1; i <= n; i++ {
		sum += turnInput(i)
	}
	return sum
}

func comma(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

func main() {
	for _, window := range []int{8_192, 1_000_000} {
		// the chat is over once ONE turn's input no longer fits the window
		n := (window-first)/growth + 1
		fmt.Printf("%10s window: full after about %s turns\n",
			comma(window), comma(n))
	}

	fmt.Printf("\n%4s %8s %14s\n", "turn", "sends", "total so far")
	for _, n := range []int{1, 2, 5, 10, 20, 30} {
		fmt.Printf("%4d %8s %14s\n", n, comma(turnInput(n)), comma(total(n)))
	}

	fmt.Printf("\nThe per-turn input grows in a straight line, so the total "+
		"grows like a square: 30 turns is 30x the messages but %dx the "+
		"input tokens of one turn.\n", total(30)/turnInput(1))
}
