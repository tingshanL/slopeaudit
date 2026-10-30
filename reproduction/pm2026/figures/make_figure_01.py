#!/usr/bin/env python3
"""Figure 1: equal total determinacy, different information (two panels).

Known-parameter values for simulation constructions A and B (matched D2 = .80 at the
common schedule t = 0..3 centered at 1.5). No fitting; values from fig1_values.py.
Drawn at final manuscript size (6.5 in wide) so that all lettering is >= 8 pt.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from fig1_values import SETTINGS, quantities

HERE = Path(__file__).resolve().parent
COL = {  # reference categorical slots 1-3 (validated all-pairs) + neutral inks
    "direct": "#2a78d6", "assist": "#eb6834", "nuisance": "#1baf7a",
    "resid": "#c3c2b7", "target": "#6f6e69",
    "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
}
FS = 9  # base font size at final size
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.edgecolor": COL["ink2"],
                     "axes.linewidth": 0.6, "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                     "xtick.labelsize": FS - 1, "ytick.labelsize": FS, "savefig.dpi": 300})

v = {k: quantities(**p) for k, p in SETTINGS.items()}


def apa(x, d=2):
    t = f"{x:.{d}f}"
    return t[1:] if t.startswith("0.") else t


def hbar(ax, y, segments, height=0.5, label_min=0.1):
    left = 0.0
    for value, color, text_color, text in segments:
        if value <= 1e-9:
            continue
        ax.barh(y, value, left=left, height=height, color=color, edgecolor="white", linewidth=1.5)
        if value >= label_min:
            ax.text(left + value / 2, y, text, ha="center", va="center", fontsize=FS, color=text_color)
        left += value


def style(ax, xlabel, title):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Design A", "Design B"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, color=COL["ink2"], fontsize=FS)
    ax.set_title(title, loc="left", fontsize=FS + 0.5, fontweight="bold", color=COL["ink"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw={"wspace": 0.35})

# Panel A: determinacy and its sources (denominator: latent-slope variance)
ax = axes[0]
for y, k in ((1, "A"), (0, "B")):
    q = v[k]
    hbar(ax, y, [(q["R_direct"], COL["direct"], "white", apa(q["R_direct"])),
                 (max(q["Delta_fit"], 0), COL["assist"], "white", apa(q["Delta_fit"]))])
style(ax, "Proportion of true-slope variance", "A  Same determinacy, $D^2$ = .80")
ax.legend(handles=[mpl.patches.Patch(color=COL["direct"], label="Within-person contrast ($R_{\\mathrm{direct}}$)"),
                   mpl.patches.Patch(color=COL["assist"], label="Model-assisted ($\\Delta_{\\mathrm{fit}}$)")],
          loc="upper left", bbox_to_anchor=(-0.02, -0.27), frameon=False, fontsize=FS - 0.5, handlelength=1.2)

# Panel B: composition of the fitted slope's variance (denominator: fitted-slope variance)
ax = axes[1]
for y, k in ((1, "A"), (0, "B")):
    q = v[k]
    hbar(ax, y, [(q["target_share"], COL["target"], "white", apa(q["target_share"])),
                 (max(q["kappa"], 0), COL["nuisance"], "white", apa(q["kappa"], 3)),
                 (q["resid_share"], COL["resid"], COL["ink"], apa(q["resid_share"]))], label_min=0.12)
ax.text(1.0, 1.36, "κ = 0", ha="right", fontsize=FS - 0.5, color=COL["ink"])
ax.text(1.0, 0.36, "κ = .145: max |r| = .38", ha="right", fontsize=FS - 0.5, color=COL["ink"])
style(ax, "Proportion of fitted-slope variance", "B  Different content")
ax.legend(handles=[mpl.patches.Patch(color=COL["target"], label="True slope"),
                   mpl.patches.Patch(color=COL["nuisance"], label="Level variation unrelated to slope (κ)"),
                   mpl.patches.Patch(color=COL["resid"], label="Measurement error")],
          loc="upper left", bbox_to_anchor=(-0.02, -0.27), frameon=False, fontsize=FS - 0.5, handlelength=1.2)

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"figure_01_example.{ext}", bbox_inches="tight", facecolor="white")
json.dump(v, open(HERE / "figure_01_values.json", "w"), indent=2, default=float)
print("wrote figure_01_example.pdf/.png (6.5 in) and figure_01_values.json")
