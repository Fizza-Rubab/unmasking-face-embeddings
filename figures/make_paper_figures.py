"""
Paper figures from eval_out CSVs (light-mode, print-oriented, CVD-safe palette):
  figures2/fig_text_retrieval_map.pdf/.png  -- grouped dot plot of mAP
  figures2/fig_naming_ablation.pdf/.png     -- naming accuracy vs vocabulary size
Run after eval_text_retrieval.py and eval_naming.py (no GPU needed).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical assignment (entity -> color, never cycled)
C = {"native": "#2a78d6", "bridged-arcface": "#1baf7a", "bridged-adaface": "#008300",
     "bridged-kprpe": "#4a3aa7", "unaligned": "#eda100", "random": "#e34948"}
PRETTY = {"native": "native (ceiling)", "bridged-arcface": "aligned ArcFace",
          "bridged-adaface": "aligned AdaFace", "bridged-kprpe": "aligned KPRPE",
          "unaligned": "unaligned", "random": "random transform."}
plt.rcParams.update({"font.size": 9,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.edgecolor": "#c9c8c2",
                     "axes.linewidth": 0.8, "figure.dpi": 200,
                     "axes.grid": True, "grid.color": "#e8e7e2",
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def fig_text_retrieval():
    rows = list(csv.DictReader(open("eval_out/text_retrieval_summary.csv")))
    # collapse random/unaligned over sources (they are per-source in the csv)
    def val(ds, tgt, kind):
        vs = [float(r["mAP"]) for r in rows if r["dataset"] == ds
              and r["target"] == tgt and (r["ranker"] == kind or
              (kind in ("unaligned", "random") and r["ranker"].startswith(kind)))]
        return np.mean(vs) if vs else np.nan

    targets = ["clip", "metaclip", "siglip"]
    tnames = ["CLIP", "MetaCLIP", "SigLIP"]
    series = ["native", "bridged-kprpe", "bridged-arcface", "bridged-adaface",
              "unaligned", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, ds, title in zip(axes, ["utk", "celeba"], ["UTKFace", "CelebA"]):
        x = np.arange(len(targets))
        for si, s in enumerate(series):
            y = [val(ds, t, s if "bridged" not in s else s) for t in targets]
            off = (si - 2.5) * 0.11
            ax.scatter(x + off, y, s=34, color=C[s], zorder=3,
                       label=PRETTY[s] if ds == "utk" else None)
            ax.vlines(x + off, 0, y, color=C[s], linewidth=2, alpha=0.55, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(tnames)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.85)
        ax.grid(axis="x", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("mean Average Precision")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=6,
               frameon=False, fontsize=7.5, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"figures2/fig_text_retrieval_map.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved figures2/fig_text_retrieval_map.[pdf|png]")


def fig_naming_ablation():
    rows = list(csv.DictReader(open("eval_out/naming_ablation.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    keep = ["native", "bridged-kprpe", "bridged-arcface", "bridged-adaface",
            "unaligned", "random"]
    ftag = "clip"                                  # representative target
    for ax, metric, title in zip(axes, ["top1", "top5"], ["Top-1", "Top-5"]):
        for s in keep:
            r = sorted(((int(x["vocab_size"]), float(x[metric])) for x in rows
                        if x["foundation"] == ftag and x["source"] == s))
            v, y = zip(*r)
            ax.plot(v, y, color=C[s], linewidth=2, marker="o", markersize=4,
                    label=PRETTY[s] if metric == "top1" else None)
        ax.set_xscale("log")
        ax.set_xticks([500, 1000, 2000, 4000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"{title} naming accuracy (CLIP)", fontsize=10)
        ax.set_xlabel("vocabulary size (names)")
        ax.set_ylim(0, 100)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("accuracy (%)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=6,
               frameon=False, fontsize=7.5, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"figures2/fig_naming_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved figures2/fig_naming_ablation.[pdf|png]")


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig_text_retrieval()
    fig_naming_ablation()
