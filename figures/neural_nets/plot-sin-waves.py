import numpy as np
import matplotlib.pyplot as plt

# Positions
p_cont = np.linspace(0, 31, 3000)

# d_model = 10 -> 5 sine channels
d_model = 30
num_freqs = d_model // 2

# Transformer-style frequency scaling
i = np.arange(num_freqs)
omegas = 1 / (10000 ** (2 * i / d_model))

# Slightly increased vertical spacing (top three more separated)
offsets = np.array([11.0, 8.0, 5, 2.5, 0.0])

fig, ax = plt.subplots(figsize=(10, 4.8))

# Continuous sines
for omega, offset in zip(omegas, offsets):
    ax.plot(p_cont, np.sin(omega * p_cont) + offset)

# Dashed vertical guide lines
for x in [7, 27]:
    ax.vlines(x, ymin=-1.8, ymax=12, linestyles="--", linewidth=1)

# Axis styling
ax.set_xticks(np.arange(0, 32, 5))
ax.set_xticks(np.arange(0, 32, 1), minor=True)
ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='x', which='minor', length=3)

for spine in ["top", "left", "right"]:
    ax.spines[spine].set_visible(False)

ax.set_xlabel("index")
ax.set_yticks([])
ax.set_xlim(-0.5, 31.5)
ax.set_ylim(-1.8, 12)

plt.tight_layout()
plt.savefig("sin-signals.pdf")
