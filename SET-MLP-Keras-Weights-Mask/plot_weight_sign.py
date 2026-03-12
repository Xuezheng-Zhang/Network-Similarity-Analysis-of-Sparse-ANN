"""
Plot positive/negative weight ratio per epoch from weight_sign_stats.json.
Uses layer="all", stage="after_training" to get one point per (source, run, epoch).
Output: results/weight_sign_ratio_by_epoch_{source}_{run}.png (one figure per run).
"""
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STATS_FILE = os.path.join(RESULTS_DIR, "weight_sign_stats.json")


def load_stats():
    """Load weight_sign_stats.json; return list of records."""
    if not os.path.isfile(STATS_FILE):
        return []
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ratio_by_epoch(records, stage="after_training"):
    """
    Filter to layer=all and given stage; return list of (source, run, epoch, pct_pos, pct_neg).
    """
    out = []
    for r in records:
        if r.get("layer") != "all" or r.get("stage") != stage:
            continue
        out.append({
            "source": r["source"],
            "run": r.get("run") if r.get("run") != "" else None,
            "epoch": r["epoch"],
            "pct_positive": r["pct_positive_of_nonz"],
            "pct_negative": r["pct_negative_of_nonz"],
        })
    return out


def plot_ratio_by_epoch():
    """Plot pct positive and pct negative (of non-zero weights) vs epoch."""
    records = load_stats()
    if not records:
        print(f"No data in {STATS_FILE}; run weight_sign_stats.py first.")
        return

    rows = get_ratio_by_epoch(records)
    if not rows:
        print("No layer=all, stage=after_training rows in stats.")
        return

    # Group by (source, run)
    by_key = defaultdict(list)
    for r in rows:
        key = (r["source"], r["run"] if r["run"] is not None else "single")
        by_key[key].append(r)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for (source, run_label), group in sorted(by_key.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        group = sorted(group, key=lambda x: x["epoch"])
        epochs = [x["epoch"] for x in group]
        pct_pos = [x["pct_positive"] for x in group]
        pct_neg = [x["pct_negative"] for x in group]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, pct_pos, linestyle="-", linewidth=2, label="Positive", color="C0")
        plt.plot(epochs, pct_neg, linestyle="-", linewidth=2, label="Negative", color="C1")
        plt.axhline(y=50, color="gray", linestyle="--", alpha=0.6)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Percentage of Positive/Negative weights", fontsize=12)
        plt.title(f"Positive/Negative Ratio by Epoch")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        safe_label = run_label if isinstance(run_label, str) else f"run_{run_label}"
        out_path = os.path.join(RESULTS_DIR, f"weight_sign_ratio_by_epoch_{source}_{safe_label}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()


if __name__ == "__main__":
    plot_ratio_by_epoch()
    print("Done.")
