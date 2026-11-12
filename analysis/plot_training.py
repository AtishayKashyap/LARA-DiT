"""Training-curve figure + summary table from scalars.csv (run parse_events.py first)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent.parent / "results"
figs = here / "figures"
df = pd.read_csv(here / "original_runs" / "scalars.csv")
w = df.pivot_table(index=["run", "step"], columns="tag", values="value").reset_index()

RUNS = ["baseline", "repa", "lara_dit", "ablation"]  # fixed order -> fixed colour (dataviz reference slots 1-4)
NAME = {"baseline": "Baseline", "repa": "REPA-style", "lara_dit": "LARA-DiT", "ablation": "All-DINOv2"}
COL = {"baseline": "#2a78d6", "repa": "#eb6834", "lara_dit": "#1baf7a", "ablation": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SMOOTH = 10  # points; each logged point is already a 100-step mean -> 1k-step window

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6})
fig = plt.figure(figsize=(11, 6.6), facecolor="#fcfcfb")
gs = fig.add_gridspec(2, 6, hspace=.45, wspace=.9)
k = lambda s: s / 1000

# (a) diffusion loss
ax = fig.add_subplot(gs[0, :3])
for r in RUNS:
    s = w[w.run == r]
    ax.plot(k(s.step), s["loss/diffusion"].rolling(SMOOTH).mean(), color=COL[r], label=NAME[r])
ax.set(xlim=(20, 200), ylim=(.275, .31), xlabel="Training step (k)", ylabel="Diffusion MSE (1k-step rolling mean)",
       title="(a) Denoising loss")
ax.legend(frameon=False, fontsize=8, labelcolor=INK)

# (b) alignment weight schedules (identical for LARA-DiT and the ablation)
ax = fig.add_subplot(gs[0, 3:])
s = w[w.run == "lara_dit"]
for tier, ls in [("early", "-"), ("mid", "--"), ("deep", ":")]:
    y = s[f"weight/lambda_{tier}"]
    ax.plot(k(s.step), y, color=INK2, ls=ls)
    ax.annotate(f"{tier}  {y.iloc[-1]:.3f}", (200, y.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK, annotation_clip=False)
r = w[w.run == "repa"]
ax.plot(k(r.step), r["weight/lambda_early"], color=COL["repa"])
ax.annotate("REPA fixed 0.5", (100, .5), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK)
ax.set(xlim=(0, 200), ylim=(0, .55), xlabel="Training step (k)", ylabel="Alignment weight λ",
       title="(b) λ schedule (LARA-DiT = All-DINOv2)")

# (c-e) alignment loss per tier
for i, tier in enumerate(["early", "mid", "deep"]):
    ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
    for r in RUNS[1:]:
        s = w[w.run == r]
        col = f"loss/align_{tier}"
        if col in s and s[col].notna().all():
            ax.plot(k(s.step), s[col].rolling(SMOOTH).mean(), color=COL[r], label=NAME[r])
    teach = {"early": "DINOv2", "mid": "CLIP vs DINOv2", "deep": "MAE vs DINOv2"}[tier]
    ax.set(xlim=(0, 200), ylim=(0, .8), xlabel="Training step (k)", title=f"({'cde'[i]}) {tier} tier: {teach}")
    if i == 0:
        ax.set_ylabel("1 − cosine (alignment loss)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
fig.savefig(figs / "training_curves.png", dpi=200, bbox_inches="tight")

# manuscript figure: per-tier alignment loss, LARA-DiT vs All-DINOv2 (same tiers + schedule, teacher differs)
fig2, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True, facecolor="white")
for ax, tier, teach in zip(axs, ["early", "mid", "deep"], ["DINOv2 vs DINOv2", "CLIP vs DINOv2", "MAE vs DINOv2"]):
    for r in ["lara_dit", "ablation"]:
        s = w[w.run == r]
        ax.plot(k(s.step), s[f"loss/align_{tier}"].rolling(SMOOTH).mean(), color=COL[r], label=NAME[r])
    ax.set(xlim=(0, 200), ylim=(0, .8), xlabel="Training step (k)", title=f"{tier} tier: {teach}")
axs[0].set_ylabel("1 − cosine (alignment loss)")
axs[0].legend(frameon=False, fontsize=8, labelcolor=INK)
fig2.savefig(figs / "alignment_lara_vs_ablation.png", dpi=200, bbox_inches="tight")

# summary table: final logged point vs mean of the last 10k steps (100 logged points)
rows = []
for r in RUNS:
    s = w[w.run == r].set_index("step")
    t = s[s.index > 190000]
    row = {"run": r, "diff_final": s["loss/diffusion"].iloc[-1], "diff_last10k_mean": t["loss/diffusion"].mean(),
           "diff_last10k_std": t["loss/diffusion"].std()}
    for tier in ["early", "mid", "deep"]:
        if f"loss/align_{tier}" in s and s[f"loss/align_{tier}"].notna().all():
            row[f"align_{tier}_last10k"] = t[f"loss/align_{tier}"].mean()
    rows.append(row)
pd.DataFrame(rows).round(6).to_csv(figs / "training_summary.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
