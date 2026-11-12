"""Paper figures: method schematic, metric comparison, qualitative grid. Writes to results/figures/."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

here = Path(__file__).resolve().parent.parent / "results" / "figures"
backup = here.parent / "original_runs"
RUNS = ["baseline", "repa", "lara_dit", "ablation"]
NAME = {"baseline": "Baseline", "repa": "REPA-style", "lara_dit": "LARA-DiT", "ablation": "All-DINOv2"}
COL = {"baseline": "#2a78d6", "repa": "#eb6834", "lara_dit": "#1baf7a", "ablation": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

# ---------- Fig: method schematic ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set(xlim=(0, 100), ylim=(-5, 46)); ax.axis("off")

def box(x, y, w, h, text, fc, ec=INK2, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")

def arrow(x0, y0, x1, y1, ls="-", c=INK2):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1, ls=ls, shrinkA=0, shrinkB=0))

# student path (left to right)
box(1, 18, 9, 10, "Image\n256×256", "#f3f2ee")
box(13, 18, 9, 10, "SD-VAE\nencoder\n(frozen)", "#f3f2ee")
box(25, 18, 9, 10, "Noisy latent\n32×32×4\n→ 64 tokens", "#f3f2ee")
arrow(10.3, 23, 12.7, 23); arrow(22.3, 23, 24.7, 23); arrow(34.3, 23, 37.2, 23)
tiers = [("Blocks 0–3", "early", "DINOv2 ViT-B/14\nCLS token", "γ = 2.0", 37.5), ("Blocks 4–7", "middle", "CLIP ViT-B/32\nimage embedding", "γ = 1.5", 54),
         ("Blocks 8–11", "deep", "MAE ViT-B/16\nmean patch token", "γ = 1.0", 70.5)]
for name, tier, teach, g, x in tiers:
    box(x, 18, 13, 10, f"DiT-B/4\n{name}", "#e3eefb", bold=True)
    box(x, 1, 13, 9.5, f"token pool → linear\n1 − cos,  λ(t)\n{g}", "#fdf1e3", fs=7.2)
    box(x, 35, 13, 9, f"{teach}\n(frozen, cached)", "#e6f5ee", fs=7.8)
    arrow(x + 6.5, 17.7, x + 6.5, 10.8)                 # student hidden -> head
    ax.plot([x + 13.3, x + 14.9, x + 14.9], [39.5, 39.5, 5.75], color=INK2, lw=1, ls="--")  # teacher target routed through the gap
    arrow(x + 14.9, 5.75, x + 13.4, 5.75, ls="-")
    ax.text(x + 0.2, 29, f"{tier} tier", fontsize=7.5, color=INK2, style="italic")
arrow(50.8, 23, 53.7, 23); arrow(67.3, 23, 70.2, 23)
box(87, 18, 12, 10, "ε prediction\nL_diff (MSE)", "#f3f2ee")
arrow(83.8, 23, 86.7, 23)
ax.text(50, 45.5, "Training only: teachers, cached targets and heads are discarded at sampling time",
        ha="center", fontsize=8, color=INK2)
ax.text(50, -4.2, "L = L_diff + Σ_k λ₀ exp(−γ_k t / T) · L_align,k ,   λ₀ = 0.5,  T = 200k",
        ha="center", fontsize=8.5, color=INK)
fig.savefig(here / "fig_method.png", dpi=220, bbox_inches="tight", facecolor="white")

# ---------- Fig: metrics ----------
m = {r: json.load(open(backup / "metrics" / f"{r}_metrics.json")) for r in RUNS}
fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
for ax, key, lab, better in [(axs[0], "fid", "FID (lower is better)", "low"), (axs[1], "is_mean", "Inception Score (higher is better)", "high")]:
    vals = [m[r][key] for r in RUNS]
    y = range(len(RUNS))[::-1]
    ax.barh(list(y), vals, color=[COL[r] for r in RUNS], height=0.6,
            xerr=[m[r]["is_std"] for r in RUNS] if key == "is_mean" else None, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.set_yticks(list(y), [NAME[r] for r in RUNS]); ax.set_xlabel(lab); ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * .02 + (m[RUNS[3 - yi]]["is_std"] if key == "is_mean" else 0), yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(vals) * 1.22)
axs[1].tick_params(labelleft=False)
fig.tight_layout(); fig.savefig(here / "fig_metrics.png", dpi=220, bbox_inches="tight", facecolor="white")

# ---------- Fig: qualitative grid (same index = same class label; initial noise NOT matched) ----------
idx = [0, 5, 10, 15, 20, 24]
fig, axs = plt.subplots(len(RUNS), len(idx), figsize=(9, 6.2))
for i, r in enumerate(RUNS):
    for j, n in enumerate(idx):
        a = axs[i, j]; a.imshow(Image.open(backup / "qualitative_samples" / r / f"{n:06d}.png")); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        if j == 0: a.set_ylabel(NAME[r], fontsize=9, color=INK)
        if i == 0: a.set_title(f"class {n % 100}", fontsize=8, color=INK2)
fig.subplots_adjust(wspace=.04, hspace=.06)
fig.savefig(here / "fig_samples.png", dpi=180, bbox_inches="tight", facecolor="white")
print("ok")
