import matplotlib.pyplot as plt
import numpy as np

# Data
states = ["Cloudy", "Rainy", "Sunny", "Thundering"]
counts = [20, 30, 45, 5]
total = sum(counts)
probs = [c / total for c in counts]

# Manual positions to control spacing
x = np.arange(len(states)) * 0.85   # <— closer together
bar_width = 0.5                     # <— thinner bars

plt.figure(figsize=(7, 4.5))
bars = plt.bar(x, counts, width=bar_width)

# Labels on top: frequency (probability)
for bar, c, p in zip(bars, counts, probs):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.8,
        f"{c} ({p:.2f})",
        ha="center",
        va="bottom"
    )

# Styling
plt.xticks(x, states, fontweight="bold")
plt.yticks([])
plt.tick_params(axis='x', length=0)
plt.grid(axis='y', linestyle='--', alpha=0.25)
plt.box(False)

plt.tight_layout()
plt.savefig('histogram.pdf')
