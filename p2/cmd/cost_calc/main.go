// Monthly cost of the triage bot per model, with prompt caching and the
// batch discount. Prices are per 1M tokens: (input, output, cache-read).
package main

import "fmt"

const (
	ticketsPerDay = 2_000
	days          = 30
	shortIn       = 142   // measured
	out           = 43    // measured
	longPrompt    = 3_000 // assumption: policy + examples, identical on every call
	batchDiscount = 0.5
)

type prices struct{ in, out, cached float64 }

var models = []struct {
	name string
	prices
}{
	{"claude-haiku-3-5", prices{0.10, 0.50, 0.01}},
	{"claude-sonnet-3-5", prices{2.00, 10.00, 0.10}},
	{"claude-opus-3-5", prices{4.00, 20.00, 0.20}},
	{"gpt-4-luna", prices{0.10, 0.50, 0.01}},
	{"gpt-4.1-terra", prices{1.00, 5.00, 0.05}},
	{"gpt-4.1-sol", prices{2.00, 10.00, 0.10}},
}

func month(p prices, tin, tout, cached int, batch bool) float64 {
	per := (float64(tin-cached)*p.in + float64(cached)*p.cached +
		float64(tout)*p.out) / 1e6
	if batch {
		per *= 1 - batchDiscount
	}
	return per * ticketsPerDay * days
}

func comma(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

func main() {
	cols := []string{"short prompt", "+3K prompt", "+3K cached", "cached+batch"}

	fmt.Printf("%s tickets a month\n\n", comma(ticketsPerDay*days))
	fmt.Printf("%-18s", "model")
	for _, h := range cols {
		fmt.Printf("%16s", h)
	}
	fmt.Println()
	for _, m := range models {
		scenarios := []float64{
			month(m.prices, shortIn, out, 0, false),
			month(m.prices, shortIn+longPrompt, out, 0, false),
			month(m.prices, shortIn+longPrompt, out, longPrompt, false),
			month(m.prices, shortIn+longPrompt, out, longPrompt, true),
		}
		fmt.Printf("%-18s", m.name)
		for _, v := range scenarios {
			fmt.Printf("%16s", fmt.Sprintf("$%.2f", v))
		}
		fmt.Println()
	}

	fmt.Printf("\ncached: the identical %s-token prefix is billed at the\n"+
		"cache-read price (cache writes are ignored here; they add a little on\n"+
		"the first call of each cache window).\n\n"+
		"batch: half price, answers within 24 hours. Fine for overnight triage,\n"+
		"not for live chat.\n", comma(longPrompt))
}
