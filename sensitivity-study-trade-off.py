import numpy as np
import matplotlib.pyplot as plt

criteria = ["Cooling", "Safety", "Volumetric", "Gravimetric", 
            "Sustainability", "Stability", "Cost"]

options = ["LH2", "SAF", "Methane", "JP-10"]

X = np.array([
    [4,3,1,2],
    [1,4,2,3],
    [1,2,3,4],
    [4,2,3,2],
    [4,3,2,1],
    [4,2,1,3],
    [2,3,1,4]
])

weights = np.array([0.31, 0.08, 0.18, 0.23, 0.05, 0.12, 0.04])

N = 10000
alpha = weights * 20

scores = np.zeros((N, len(options)))
rankings = np.zeros((N, len(options)))

for k in range(N):
    w = np.random.dirichlet(alpha)
    s = w @ X
    scores[k] = s
    rankings[k] = np.argsort(-s) 



best_counts = np.zeros(len(options))
for r in rankings:
    best_counts[int(r[0])] += 1

prob_best = best_counts / N


# ---------------------------
# Reporting
# ---------------------------

plt.style.use("default")

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

# Soft gray backgrounds
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

# Clear, colorblind-friendly palette
colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728"   # red
]

for i, f in enumerate(options):
    ax.hist(
        scores[:, i],
        bins=60,
        alpha=0.45,
        label=f,
        color=colors[i % len(colors)],
        edgecolor="#222222",
        linewidth=0.5
    )

ax.set_title(
    "Fuel Score Distributions",
    fontsize=14,
    pad=12
)

ax.set_xlabel("Trade-off Score", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)



# Cleaner grid
ax.grid(
    alpha=0.25,
    linestyle="--",
    linewidth=0.7
)

# Slightly transparent legend
legend = ax.legend(frameon=True)
legend.get_frame().set_alpha(0.9)
legend.get_frame().set_facecolor("#FFFFFF")  # legend background
legend.get_frame().set_edgecolor("#000000")  # border color

plt.tight_layout()
plt.show()

# ---------------------------
# Print ranking robustness
# ---------------------------
for i, f in enumerate(options):
    print(f"{f}: P(best) = {prob_best[i]:.3f}")