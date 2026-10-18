"""Rebuild the charts in assets/ (used by README.md) from runs/ and results/. Run: python make_figs.py"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
MUTED, INK, INK2, GRID, SURFACE = "#b5b4ae", "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titlecolor": INK, "font.size": 9,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote assets/" + name)


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


# 1. Loss curves, one panel per training run
runs = sorted((ROOT / "runs").glob("*_loss.jsonl"))
if runs:
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 2.8), squeeze=False)
    for ax, path in zip(axes[0], runs):
        recs = read_jsonl(path)
        cfg = recs[0]
        tr = [r for r in recs if r["type"] == "train"]
        va = [r for r in recs if r["type"] == "val"]
        ax.plot([r["epoch"] for r in tr], [r["loss"] for r in tr], color=BLUE, lw=1.2, label="train")
        ax.plot([r["epoch"] for r in va], [r["loss"] for r in va], color=ORANGE, lw=2,
                marker="o", ms=4, label="validation")
        tr_end = np.mean([r["loss"] for r in tr[-5:]])   # smooth the noisy per-step train loss
        ax.text(0.03, 0.05, f"end: train {tr_end:.1g} · val {va[-1]['loss']:.1g}  "
                f"(gap {va[-1]['loss'] / tr_end:.0f}×)", transform=ax.transAxes, color=INK2, fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss on answer tokens (log)")
        n = cfg["n_train"] or "all"
        ax.set_title(f"run '{cfg['run']}': {n} train examples, {cfg['epochs']} epochs")
        ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "loss_curves.png")

# 2 + 3. Accuracy charts from results/summary.jsonl (newest row per split + method)
METHODS = [  # (tag in summary.jsonl, label, fine-tuned?)
    ("zero_shot", "0.5B zero-shot", False),
    ("few_shot", "0.5B few-shot (3 examples)", False),
    ("ollama-qwen2.5-1.5b", "Qwen 1.5B zero-shot", False),
    ("ollama-llama3.2-3b", "Llama 3B zero-shot", False),
    ("lora-overfit-epoch8", "0.5B + LoRA, 40 ex × 8 ep", True),
    ("lora-main-epoch2", "0.5B + LoRA, 300 ex × 2 ep", True),
    ("lora-main-epoch1", "0.5B + LoRA, 300 ex × 1 ep", True),
]
TITLES = {"test_id": "Same templates & organisms as training",
          "test_ood": "Unseen templates & unseen organisms"}
summ_path = ROOT / "results" / "summary.jsonl"
if summ_path.exists():
    latest = {(r["split"], r["tag"]): r for r in read_jsonl(summ_path)}

    # 2. Headline: share of examples with all three fields right, one panel per test set
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    for ax, split in zip(axes, ["test_id", "test_ood"]):
        rows = [(lab, latest[(split, t)]["all_fields"], ft) for t, lab, ft in METHODS if (split, t) in latest]
        y = np.arange(len(rows))
        bars = ax.barh(y, [v for _, v, _ in rows], height=0.62,
                       color=[AQUA if ft else BLUE for *_, ft in rows])
        for b, (_, v, _) in zip(bars, rows):
            ax.annotate(f"{v:.0%}", (v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", color=INK2, fontsize=8)
        ax.set_yticks(y, [lab for lab, _, _ in rows])
        ax.set_xlim(0, 1.12)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", visible=False)
        ax.set_title(TITLES[split])
        ax.set_xlabel("all 3 fields correct")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, AQUA)]
    fig.legend(handles, ["prompted", "fine-tuned (LoRA)"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.6, -0.06))
    fig.tight_layout()
    save(fig, "accuracy.png")

    # 3. Per-field breakdown on the unseen-organism test set, for the four methods worth comparing
    pick = [("few_shot", "0.5B few-shot", BLUE), ("ollama-llama3.2-3b", "Llama 3B zero-shot", ORANGE),
            ("lora-main-epoch1", "LoRA 300 ex", AQUA), ("lora-overfit-epoch8", "LoRA 40 ex", YELLOW)]
    pick = [p for p in pick if ("test_ood", p[0]) in latest]
    fields = [("organism", "organism"), ("assay", "assay"), ("sample_size", "sample size"),
              ("all_fields", "all 3 correct")]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    x = np.arange(len(fields))
    w = 0.8 / len(pick)
    for i, (t, lab, c) in enumerate(pick):
        vals = [latest[("test_ood", t)][f] for f, _ in fields]
        ax.bar(x + (i - (len(pick) - 1) / 2) * w, vals, w - 0.02, color=c, label=lab)
    ax.set_xticks(x, [n for _, n in fields])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    ax.set_title("Unseen organisms: fine-tuning keeps the skills, loses the labels")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fields_ood.png")
