import numpy as np
import matplotlib.pyplot as plt

# Positions 0 to 31
p = np.arange(32)

# Binary bits (LSB to 4th bit)
bits = [(p >> i) & 1 for i in range(5)]

# Vertical offsets
offsets = [6.0, 4.5, 3.0, 1.5, 0.0]

fig, ax = plt.subplots(figsize=(10, 4))

for bit, offset in zip(bits, offsets):
    ax.step(p, bit + offset, where="post")

# Dashed lines: extend clearly below the x-axis
y_bottom = -1.8   # noticeably below x-axis
y_top = 7.6

for x in [7, 27]:
    ax.vlines(x, ymin=y_bottom, ymax=y_top, linestyles="--", linewidth=1)

# Bit-reversed annotation well below the x-axis
def annotate_bits_reversed_below(x):
    b = [(x >> i) & 1 for i in range(5)]  # LSB → MSB
    label = "".join(str(bit) for bit in b)
    ax.text(x, y_bottom - 0.3, label, ha="center", va="top")

annotate_bits_reversed_below(7)
annotate_bits_reversed_below(27)

# X-axis ticks
ax.set_xticks(np.arange(0, 32, 5))
ax.set_xticks(np.arange(0, 32, 1), minor=True)
ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='x', which='minor', length=3)

# Style
for spine in ["top", "left", "right"]:
    ax.spines[spine].set_visible(False)

ax.set_xlabel("index")
ax.set_yticks([])
ax.set_ylim(-1.8, 7.8)
ax.set_xlim(-0.5, 31.5)

plt.tight_layout()
plt.savefig("bin-signals.pdf")