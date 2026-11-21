"""Every turn resends the whole conversation, so the input of each turn
grows in a straight line and the running total grows like a square."""
FIRST, GROWTH = 132, 260  # first-turn input tokens, extra tokens per turn


def turn_input(n):
    return FIRST + (n - 1) * GROWTH


def total(n):
    return sum(turn_input(i) for i in range(1, n + 1))


for window in (8_192, 1_000_000):
    # the chat is over once ONE turn's input no longer fits the window
    n = (window - FIRST) // GROWTH + 1
    print(f"{window:>10,} window: full after about {n:,} turns")

print(f"\n{'turn':>4} {'sends':>8} {'total so far':>14}")
for n in (1, 2, 5, 10, 20, 30):
    print(f"{n:>4} {turn_input(n):>8,} {total(n):>14,}")

print("\nThe per-turn input grows in a straight line, so the total grows "
      f"like a square: 30 turns is 30x the messages but "
      f"{total(30) // turn_input(1)}x the input tokens of one turn.")
