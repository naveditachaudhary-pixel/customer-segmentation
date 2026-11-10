"""README charts (light + dark) from results/*.csv.

Run from the repo root after segment.py:  python src/make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, DOCS = ROOT / "results", ROOT / "docs"
ORDER = ["Big-basket loyalists", "Fuel & store-brand regulars", "Occasional shoppers", "Lapsed"]

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", ref="#8a8984",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#383835", ref="#8a8984",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}


def style(ax, t, grid_axis="y"):
    ax.set_facecolor(t["bg"])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)


def k_chart(mode):
    t = THEMES[mode]
    k = pd.read_csv(RES / "k_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), facecolor=t["bg"])
    for ax, col, title, fmt in [
        (axes[0], "silhouette", "Silhouette (higher = better separated)", "{:.2f}"),
        (axes[1], "stability_ari", "Stability across random starts (ARI)", "{:.2f}"),
    ]:
        style(ax, t)
        ax.plot(k.k, k[col], color=t["series"][0], lw=2, marker="o", ms=6,
                markeredgecolor=t["bg"], markeredgewidth=2)
        sel = k[k.k == 4].iloc[0]
        ax.plot([4], [sel[col]], marker="o", ms=10, color=t["series"][1],
                markeredgecolor=t["bg"], markeredgewidth=2, zorder=5)
        ax.annotate(f"k = 4: {fmt.format(sel[col])}", (4, sel[col]), xytext=(8, -14),
                    textcoords="offset points", fontsize=9, color=t["ink"])
        ax.set_title(title, loc="left", fontsize=10, color=t["ink"], fontweight="bold")
        ax.set_xticks(k.k)
        ax.set_xlabel("number of segments (k)", fontsize=9, color=t["ink2"])
        ax.set_ylim(0, 1.05 if col == "stability_ari" else 0.3)
    fig.tight_layout()
    fig.savefig(DOCS / f"k_selection_{mode}.png", dpi=160, facecolor=t["bg"])
    plt.close(fig)


def pca_chart(mode):
    t = THEMES[mode]
    f = pd.read_csv(RES / "household_segments.csv")
    fig, ax = plt.subplots(figsize=(7.6, 5.6), facecolor=t["bg"])
    style(ax, t, grid_axis="both")
    for name, color in zip(ORDER, t["series"]):
        d = f[f.segment == name]
        ax.scatter(d.pc1, d.pc2, s=9, color=color, alpha=0.55, lw=0, label=f"{name} ({len(d):,})")
    for name in ORDER:  # direct labels at the segment centre
        d = f[f.segment == name]
        ax.text(d.pc1.median(), d.pc2.median(), name, fontsize=9, fontweight="bold",
                color=t["ink"], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=t["bg"], ec="none", alpha=0.85))
    ax.set_xlabel("PC1: overall engagement (trips, spend, departments)", fontsize=9, color=t["ink2"])
    ax.set_ylabel("PC2: store brands, fuel and promotions", fontsize=9, color=t["ink2"])
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, markerscale=2.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.suptitle("2,469 households, four segments (PCA view)", x=0.08, ha="left",
                 color=t["ink"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS / f"pca_segments_{mode}.png", dpi=160, facecolor=t["bg"])
    plt.close(fig)


def share_chart(mode):
    t = THEMES[mode]
    p = pd.read_csv(RES / "segment_profiles.csv", index_col=0).reindex(ORDER)
    fig, ax = plt.subplots(figsize=(7.6, 3.6), facecolor=t["bg"])
    style(ax, t, grid_axis="x")
    y = range(len(ORDER))[::-1]
    h = 0.34
    for yi, (name, r) in zip(y, p.iterrows()):
        for off, val, color in [(h / 2 + 0.02, r.share_of_households, t["series"][0]),
                                (-h / 2 - 0.02, r.share_of_spend, t["series"][1])]:
            ax.barh(yi + off, val, height=h, color=color, lw=0)
            ax.text(val + 0.008, yi + off, f"{val:.0%}", va="center", fontsize=8.5, color=t["ink"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(ORDER, fontsize=9, color=t["ink"])
    ax.set_xlim(0, 0.7)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    from matplotlib.patches import Patch
    leg = ax.legend(handles=[Patch(color=t["series"][0], label="share of households"),
                             Patch(color=t["series"][1], label="share of spend")],
                    loc="lower right", frameon=False, fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.suptitle("Half the households bring in 85% of the spend", x=0.03, ha="left",
                 color=t["ink"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS / f"segment_share_{mode}.png", dpi=160, facecolor=t["bg"])
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    for mode in THEMES:
        k_chart(mode)
        pca_chart(mode)
        share_chart(mode)
