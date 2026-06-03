"""Generate README figures from the project's numbers. Run: python docs/figures/make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#444", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
})
INK, MUTED = "#22303f", "#6b7a8d"
TEAL, CORAL, GOLD = "#2a9d8f", "#e76f51", "#e9c46a"


# ---- Figure 1: orientation accuracy by data type ----
fig, ax = plt.subplots(figsize=(7.6, 4.6))
labels = ["DREAM4\n(time-series,\nsimulated)", "BEELINE\n(static\nsingle-cell)", "RPE1\n(CRISPR\ninterventions)"]
centers = [0.92, 0.60, 0.67]
lo = [0.88, 0.50, 0.61]; hi = [0.96, 1.00, 0.70]
x = np.arange(3)
ax.bar(x, centers, width=0.55, color=[TEAL, GOLD, CORAL], edgecolor=INK, linewidth=0.8, zorder=3)
ax.errorbar(x, centers, yerr=[np.array(centers) - lo, np.array(hi) - centers],
            fmt="none", ecolor=INK, elinewidth=1.3, capsize=6, zorder=4)
ax.axhline(0.5, ls="--", color=MUTED, lw=1.1, zorder=2)
ax.text(2.55, 0.47, "0.5 = random", color=MUTED, fontsize=9, va="top", ha="right")
for xi, c in zip(x, centers):
    ax.text(xi, c + 0.05, f"{c:.2f}", ha="center", fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlim(-0.75, 2.6); ax.set_ylim(0.4, 1.05)
ax.set_ylabel("orientation accuracy")
ax.set_title("Edge-direction recovery by data type")
fig.tight_layout(); fig.savefig(OUT / "fig1_regime_ladder.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---- Figure 2: synthetic vs real performance ----
fig, ax = plt.subplots(figsize=(7.6, 4.8))
ideas = ["direct vs chain\n(3rd-gene)", "factor separation\n(overfitting)", "wiring recovery\n(inverse)"]
clean = [0.98, 1.00, 0.98]; real = [0.50, 0.48, 0.50]
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w/2, clean, w, color=TEAL, edgecolor=INK, linewidth=0.8, zorder=3)
b2 = ax.bar(x + w/2, real, w, color="#c2c7cd", edgecolor=INK, linewidth=0.8, zorder=3)
ax.axhline(0.5, ls="--", color=MUTED, lw=1.1, zorder=2)
for xi, c, r in zip(x, clean, real):
    ax.text(xi - w/2, c + 0.02, f"{c:.2f}", ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(xi + w/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels(ideas)
ax.set_ylim(0, 1.15); ax.set_ylabel("performance (0.5 = random)")
ax.set_title("Synthetic vs real data, three methods")
ax.legend([b1, b2], ["synthetic data", "RPE1 data"], frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig2_clean_vs_real.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---- Figure 3: dominant response component ----
fig, ax = plt.subplots(figsize=(6.4, 4.2), subplot_kw=dict(aspect="equal"))
ax.pie([53, 47], colors=[CORAL, "#e7e9ec"], startangle=90, counterclock=False,
       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
ax.text(0, 0.10, "53%", ha="center", fontsize=25, fontweight="bold", color=CORAL)
ax.text(0, -0.18, "top component", ha="center", fontsize=10, color=MUTED)
ax.set_title("Dominant component of the RPE1 response matrix", pad=12)
fig.tight_layout(); fig.savefig(OUT / "fig3_cascade.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("wrote fig1_regime_ladder.png, fig2_clean_vs_real.png, fig3_cascade.png")
