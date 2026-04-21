
import json
import numpy as np
import os
import sys
from scipy.sparse import load_npz

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


SOURCES = {
    "set_mlp": os.path.join(RESULTS_DIR, "graph_snapshots"),
    "rbm": os.path.join(RESULTS_DIR, "graph_snapshots_rbm"),
    "static": os.path.join(RESULTS_DIR, "graph_snapshots_static"),
}


def load_weights_from_directory(stage_dir):

    weights = {}
    available_layers = []
    for layer in [1, 2, 3, 4]:
        path = os.path.join(stage_dir, f"weight_layer_{layer}.npz")
        if os.path.exists(path):
            w = load_npz(path)
            weights[f"layer_{layer}"] = w.toarray()
            available_layers.append(layer)
    return weights, available_layers


def stats_one_matrix(W):

    flat = np.ravel(W)
    n_total = flat.size
    n_positive = int(np.sum(flat > 0))
    n_negative = int(np.sum(flat < 0))
    n_zero = int(n_total - n_positive - n_negative)
    n_nonz = n_positive + n_negative
    if n_nonz == 0:
        pct_pos = pct_neg = 0.0
    else:
        pct_pos = 100.0 * n_positive / n_nonz
        pct_neg = 100.0 * n_negative / n_nonz
    pct_zero = 100.0 * n_zero / n_total if n_total else 0.0
    return {
        "n_total": n_total,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "n_nonz": n_nonz,
        "pct_positive_of_nonz": pct_pos,
        "pct_negative_of_nonz": pct_neg,
        "pct_zero": pct_zero,
    }


def collect_stats_for_stage(stage_dir):

    weights, available_layers = load_weights_from_directory(stage_dir)
    if not available_layers:
        return []

    rows = []
    all_pos = all_neg = all_zero = 0
    for layer in sorted(available_layers):
        W = weights[f"layer_{layer}"]
        s = stats_one_matrix(W)
        rows.append({"layer": f"layer_{layer}", **s})
        all_pos += s["n_positive"]
        all_neg += s["n_negative"]
        all_zero += s["n_zero"]

    n_nonz = all_pos + all_neg
    if n_nonz > 0:
        pct_pos = 100.0 * all_pos / n_nonz
        pct_neg = 100.0 * all_neg / n_nonz
    else:
        pct_pos = pct_neg = 0.0
    n_total = all_pos + all_neg + all_zero
    rows.append({
        "layer": "all",
        "n_total": n_total,
        "n_positive": all_pos,
        "n_negative": all_neg,
        "n_zero": all_zero,
        "n_nonz": n_nonz,
        "pct_positive_of_nonz": pct_pos,
        "pct_negative_of_nonz": pct_neg,
        "pct_zero": 100.0 * all_zero / n_total if n_total else 0.0,
    })
    return rows


def process_snapshots_dir(input_dir, source_name, run_id=None):

    epoch_dirs = sorted(
        [d for d in os.listdir(input_dir)
         if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1]),
    )
    results = []
    for epoch_dir_name in epoch_dirs:
        epoch_num = int(epoch_dir_name.split("_")[1])
        epoch_path = os.path.join(input_dir, epoch_dir_name)
        for stage in ("after_training", "after_pruning"):
            stage_dir = os.path.join(epoch_path, stage)
            if not os.path.isdir(stage_dir):
                continue
            for row in collect_stats_for_stage(stage_dir):
                results.append({
                    "source": source_name,
                    "run": run_id if run_id is not None else "",
                    "epoch": epoch_num,
                    "stage": stage,
                    "layer": row["layer"],
                    "n_total": row["n_total"],
                    "n_positive": row["n_positive"],
                    "n_negative": row["n_negative"],
                    "n_zero": row["n_zero"],
                    "n_nonz": row["n_nonz"],
                    "pct_positive_of_nonz": round(row["pct_positive_of_nonz"], 4),
                    "pct_negative_of_nonz": round(row["pct_negative_of_nonz"], 4),
                    "pct_zero": round(row["pct_zero"], 4),
                })
    return results


def run_source(source_name, input_base):

    if not os.path.isdir(input_base):
        print(f"Skipping {source_name}: directory not found {input_base}")
        return []

    run_dirs = sorted(
        [d for d in os.listdir(input_base)
         if os.path.isdir(os.path.join(input_base, d)) and d.startswith("run_")],
        key=lambda x: int(x.replace("run_", "")),
    )
    all_results = []
    if run_dirs:
        for run_name in run_dirs:
            run_id = int(run_name.replace("run_", ""))
            run_path = os.path.join(input_base, run_name)
            print(f"  {source_name} / {run_name} ...")
            all_results.extend(process_snapshots_dir(run_path, source_name, run_id=run_id))
    else:
        print(f"  {source_name} (single model) ...")
        all_results.extend(process_snapshots_dir(input_base, source_name, run_id=None))
    return all_results


def write_json(results, out_path):

    if not results:
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():

    source_arg = (sys.argv[1] if len(sys.argv) > 1 else "set_mlp").strip().lower()
    if source_arg not in ("set_mlp", "rbm", "static", "all"):
        print("Usage: python weight_sign_stats.py [set_mlp|rbm|static|all]")
        print("Default: set_mlp")
        source_arg = "set_mlp"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "weight_sign_stats.json")

    all_results = []
    if source_arg == "all":
        for name, input_dir in SOURCES.items():
            print(f"\n[{name}]")
            all_results.extend(run_source(name, input_dir))
    else:
        input_dir = SOURCES[source_arg]
        print(f"\n[{source_arg}]")
        all_results.extend(run_source(source_arg, input_dir))

    if not all_results:
        print("No weight data found; run training and save graph_snapshots first.")
        return

    write_json(all_results, json_path)
    print(f"\nResults written to: {json_path}")


if __name__ == "__main__":
    main()
