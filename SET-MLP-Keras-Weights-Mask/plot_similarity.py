import json
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np
import argparse

RESULTS_DIR = "SET-MLP-Keras-Weights-Mask/results"
SIMILARITY_DIR = os.path.join(RESULTS_DIR, "similarity")
GRAPH_DIR = os.path.join(RESULTS_DIR, "diagram")
TARGET_STEP_SIZES_BY_SOURCE = {
    "set_mlp": [3, 5, 10, 20],
    "rbm": [1, 2],
    "static": [3, 5, 10, 20],
}

SIMILARITY_COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
STAGE_ORDER = ["Initial", "Training", "Stabilization"]
STAGE_LABELS = {
    "Initial": "Initial (10%-50%)",
    "Training": "Training (50%-80%)",
    "Stabilization": "Stabilization (80%+)",
}

VALID_SOURCES = ("set_mlp", "rbm", "static")
VALID_MATRIX_TYPES = ("binary", "weighted", "dual", "all")


def _resolve_similarity_file(filename, base_dir=RESULTS_DIR):
    candidates = [os.path.join(base_dir, filename), os.path.join(base_dir, "similarity", filename)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]

def load_val_accuracy_data(source="set_mlp"):


    if source == "rbm":
        pat = os.path.join(RESULTS_DIR, "training_metadata_rbm_run_*.json")
    elif source == "static":
        pat = os.path.join(RESULTS_DIR, "training_metadata_static_run_*.json")
    else:
        pat = os.path.join(RESULTS_DIR, "training_metadata_run_*.json")

    files = sorted(glob.glob(pat))
    if files:
        acc_by_epoch = {}
        for p in files:
            with open(p) as f:
                recs = json.load(f)
            for r in recs:
                e = r["epoch"]
                if e not in acc_by_epoch:
                    acc_by_epoch[e] = []
                acc_by_epoch[e].append(r["val_accuracy"])
        return {e: np.mean(a) for e, a in acc_by_epoch.items()}


    path = os.path.join(RESULTS_DIR, "training_metadata.json")
    if os.path.isfile(path):
        with open(path) as f:
            recs = json.load(f)
        return {r["epoch"]: r["val_accuracy"] for r in recs}
    return {}

def aggregate_similarity_mean(df):

    between = df[df["Type"] == "Between_Epochs"].copy()
    if between.empty:
        return between
    if "n" not in between.columns:
        between["n"] = between["Epoch2"] - between["Epoch1"]
    if "Run" not in between.columns:
        return between
    agg = between.groupby(["Epoch1", "Epoch2", "n"], as_index=False)["Similarity"].mean()
    return agg

def get_val_accuracy_range(epoch1, epoch2, epoch_to_acc):
    accuracies = []
    if epoch1 in epoch_to_acc:
        accuracies.append(epoch_to_acc[epoch1])
    if epoch2 in epoch_to_acc:
        accuracies.append(epoch_to_acc[epoch2])

    if not accuracies:
        return None

    return accuracies


def add_accuracy_line(epoch_to_acc):

    if not epoch_to_acc:
        return
    ax = plt.gca()
    ax.set_ylabel("Similarity Score", fontsize=12, color="black")
    ax.tick_params(axis="y", labelcolor="black")
    ax.spines["left"].set_color("black")
    ax2 = ax.twinx()
    ep = sorted(epoch_to_acc.keys())
    acc = [epoch_to_acc[e] for e in ep]
    ax2.plot(ep, acc, color="green", linewidth=2, label="Val Accuracy")
    ax2.set_ylabel("Val Accuracy", fontsize=12, color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.spines["right"].set_color("green")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")


def _set_xlabel_rotation_30():

    fig = plt.gcf()
    if not fig.axes:
        return
    main_ax = fig.axes[0]
    for label in main_ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def _plot_similarity_multi_n(df, epoch_to_acc, title, output_path, source="set_mlp"):

    between_epochs = aggregate_similarity_mean(df)
    if between_epochs.empty:
        print(f"No Between_Epochs data for {title}; skip.")
        return

    available_n = set(between_epochs["n"].tolist())
    preferred_steps = TARGET_STEP_SIZES_BY_SOURCE.get(source, TARGET_STEP_SIZES_BY_SOURCE["set_mlp"])
    n_values = [n for n in preferred_steps if n in available_n]
    if not n_values:

        n_values = sorted(available_n)
        print(
            f"No preferred step sizes {preferred_steps} found for {title}. "
            f"Fallback to available n: {n_values}"
        )

    missing_n = [n for n in preferred_steps if n not in available_n]
    if missing_n:
        print(f"Missing step sizes for {title}: {missing_n}")

    plt.figure(figsize=(12, 8))
    ax_left = plt.gca()

    for idx, n_val in enumerate(n_values):
        data_n = between_epochs[between_epochs["n"] == n_val].copy()
        data_n = data_n.sort_values("Epoch1")
        epochs = data_n["Epoch1"].values
        similarities = data_n["Similarity"].values
        ax_left.plot(
            epochs,
            similarities,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            color=SIMILARITY_COLORS[idx],
            label=f"n={n_val}",
        )

    add_accuracy_line(epoch_to_acc)
    ax_left = plt.gcf().axes[0]
    ax_left.set_title(title, fontsize=14, fontweight="bold", color="black")
    ax_left.set_xlabel("Epoch", fontsize=12, color="black")
    ax_left.set_ylabel("Similarity Score", fontsize=12, color="black")
    ax_left.tick_params(axis="y", labelcolor="black")
    ax_left.spines["left"].set_color("black")
    ax_left.grid(True, which="both", linestyle="--", alpha=0.5)

    if not between_epochs.empty:
        max_epoch = int(between_epochs["Epoch1"].max())
        xticks = list(range(0, max_epoch + 1, 10))
        if xticks and xticks[-1] != max_epoch:
            xticks.append(max_epoch)
        ax_left.set_xticks(xticks)

    _set_xlabel_rotation_30()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_deltacon_similarity_by_n(matrix_type="binary", source="set_mlp"):

    if matrix_type == "dual":
        plot_deltacon_similarity_dual_by_n(source)
        return

    if source == "set_mlp":
        similarity_file = _resolve_similarity_file(f"deltacon_similarity_{matrix_type}.csv")
    else:
        similarity_file = _resolve_similarity_file(f"deltacon_similarity_{source}_{matrix_type}.csv")
    if not os.path.isfile(similarity_file):
        print(f"No {similarity_file}; skip DeltaCon ({source}, {matrix_type}) plot.")
        return
    df = pd.read_csv(similarity_file)
    epoch_to_acc = load_val_accuracy_data(source)
    output_dir = GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    source_prefix = "" if source == "set_mlp" else f"{source}_"
    name_suffix = "" if matrix_type == "binary" else f"_{matrix_type}"
    output_path = os.path.join(output_dir, f"{source_prefix}deltacon_similarity{name_suffix}.png")
    _plot_similarity_multi_n(df, epoch_to_acc, "DeltaCon Similarity", output_path, source=source)

def plot_deltacon_similarity_dual_by_n(source="set_mlp"):

    if source == "set_mlp":
        similarity_file = _resolve_similarity_file("deltacon_similarity_set_mlp_dual.csv")
    else:
        similarity_file = _resolve_similarity_file(f"deltacon_similarity_{source}_dual.csv")
    if not os.path.isfile(similarity_file):
        print(f"No {similarity_file}; skip DeltaCon ({source}, dual) plot.")
        return
    df = pd.read_csv(similarity_file)
    epoch_to_acc = load_val_accuracy_data(source)
    output_dir = GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    source_prefix = "" if source == "set_mlp" else f"{source}_"
    output_path = os.path.join(output_dir, f"{source_prefix}deltacon_similarity_dual.png")
    _plot_similarity_multi_n(df, epoch_to_acc, "Dual DeltaCon Similarity", output_path, source=source)

def plot_jaccard_similarity_by_n(matrix_type="binary", source="set_mlp"):

    if matrix_type == "dual":
        plot_jaccard_similarity_dual_by_n(source)
        return
    if source == "set_mlp":
        similarity_file = _resolve_similarity_file(f"jaccard_similarity_{matrix_type}.csv")
    else:
        similarity_file = _resolve_similarity_file(f"jaccard_similarity_{source}_{matrix_type}.csv")
    if not os.path.isfile(similarity_file):
        print(f"No {similarity_file}; skip Jaccard ({source}, {matrix_type}) plot.")
        return
    df = pd.read_csv(similarity_file)
    epoch_to_acc = load_val_accuracy_data(source)
    output_dir = GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    source_prefix = "" if source == "set_mlp" else f"{source}_"
    name_suffix = "" if matrix_type == "binary" else f"_{matrix_type}"
    output_path = os.path.join(output_dir, f"{source_prefix}jaccard_similarity{name_suffix}.png")
    _plot_similarity_multi_n(df, epoch_to_acc, "Jaccard Similarity", output_path, source=source)

def plot_jaccard_similarity_dual_by_n(source="set_mlp"):

    if source == "set_mlp":
        similarity_file = _resolve_similarity_file("jaccard_similarity_set_mlp_dual.csv")
    else:
        similarity_file = _resolve_similarity_file(f"jaccard_similarity_{source}_dual.csv")
    if not os.path.isfile(similarity_file):
        print(f"No {similarity_file}; skip Jaccard ({source}, dual) plot.")
        return
    df = pd.read_csv(similarity_file)
    epoch_to_acc = load_val_accuracy_data(source)
    output_dir = GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    source_prefix = "" if source == "set_mlp" else f"{source}_"
    output_path = os.path.join(output_dir, f"{source_prefix}jaccard_similarity_dual.png")
    _plot_similarity_multi_n(df, epoch_to_acc, "Jaccard Similarity", output_path, source=source)

def plot_similarity_between_epochs():
    similarity_file = "SET-MLP-Keras-Weights-Mask/results/deltacon_similarity_binary.csv"
    df = pd.read_csv(similarity_file)
    between_epochs = df[df['Type'] == 'Between_Epochs'].copy()


    between_epochs = between_epochs.sort_values(['Epoch1', 'Epoch2'])

    epochs = between_epochs['Epoch1'].values
    similarities = between_epochs['Similarity'].values

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, similarities, linestyle='-', color='b', linewidth=2)

    plt.title('DeltaCon Similarity Evolution', fontsize=14)
    plt.xlabel('Epoch Transition (t to t+1)', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(epochs, rotation=30)

    os.makedirs(GRAPH_DIR, exist_ok=True)
    output_path = os.path.join(GRAPH_DIR, "similarity_between_epochs.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Between_Epochs plot saved to: {output_path}")
    plt.close()

def plot_similarity_within_epochs():
    similarity_file = "SET-MLP-Keras-Weights-Mask/results/deltacon_similarity_binary.csv"
    df = pd.read_csv(similarity_file)
    within_epochs = df[df['Type'] == 'Within_Epoch'].copy()


    within_epochs = within_epochs.sort_values(['Epoch1', 'Epoch2'])

    epochs = within_epochs['Epoch1'].values
    similarities = within_epochs['Similarity'].values

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, similarities, linestyle='-', color='r', linewidth=2)

    plt.title('DeltaCon Similarity Evolution', fontsize=14)
    plt.xlabel('Epoch Transition (t to t+n)', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(epochs, rotation=30)

    os.makedirs(GRAPH_DIR, exist_ok=True)
    output_path = os.path.join(GRAPH_DIR, "similarity_within_epochs.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Within_Epochs plot saved to: {output_path}")
    plt.close()


def _parse_sparsities(sparsities_arg):
    return [int(x.strip()) for x in sparsities_arg.split(",") if x.strip()]


def load_val_accuracy_data_from_metadata(metadata_path):
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path) as f:
        recs = json.load(f)
    return {r["epoch"]: r["val_accuracy"] for r in recs if "epoch" in r and "val_accuracy" in r}


def map_phase_from_accuracy(acc):
    if pd.isna(acc):
        return None
    if 0.10 <= acc < 0.50:
        return "Initial"
    if 0.50 <= acc < 0.80:
        return "Training"
    if acc >= 0.80:
        return "Stabilization"
    return None


def build_phase_similarity_summary(experiment_root, metric="deltacon", matrix_type="dual", n=10, sparsities=None):
    if sparsities is None:
        sparsities = [50, 70, 90, 98]
    combined_csv = _resolve_similarity_file(
        f"{metric}_similarity_set_mlp_sparsities_{matrix_type}_n{n}.csv",
        base_dir=experiment_root,
    )
    if not os.path.isfile(combined_csv):
        raise FileNotFoundError(f"Missing combined similarity CSV: {combined_csv}")
    df = pd.read_csv(combined_csv)
    if df.empty:
        raise ValueError(f"Combined similarity CSV is empty: {combined_csv}")

    between = df[df["Type"] == "Between_Epochs"].copy()
    if "n" in between.columns:
        between = between[between["n"] == n].copy()
    if between.empty:
        raise ValueError(f"No Between_Epochs rows for n={n} in {combined_csv}")

    phase_rows = []
    for sparsity in sparsities:
        sub = between[between["Sparsity"] == sparsity].copy()
        if sub.empty:
            continue
        metadata_path = os.path.join(
            experiment_root, f"set_mlp_s{sparsity}", "training_metadata_run_0.json"
        )
        epoch_to_acc = load_val_accuracy_data_from_metadata(metadata_path)
        if not epoch_to_acc:
            print(f"Warning: missing val_accuracy metadata for sparsity={sparsity}: {metadata_path}")
            continue
        sub["Val_Accuracy_Epoch1"] = sub["Epoch1"].map(epoch_to_acc)
        sub["Stage"] = sub["Val_Accuracy_Epoch1"].apply(map_phase_from_accuracy)
        sub = sub[sub["Stage"].isin(STAGE_ORDER)].copy()
        phase_rows.append(sub)

    if not phase_rows:
        raise ValueError("No usable rows after phase mapping. Check metadata and epoch alignment.")

    phase_df = pd.concat(phase_rows, ignore_index=True)
    summary = (
        phase_df.groupby(["Sparsity", "Stage"], as_index=False)["Similarity"]
        .agg(Avg_Similarity="mean", Count="count")
    )


    all_pairs = pd.MultiIndex.from_product(
        [sorted(sparsities), STAGE_ORDER], names=["Sparsity", "Stage"]
    ).to_frame(index=False)
    summary = all_pairs.merge(summary, on=["Sparsity", "Stage"], how="left")
    summary["Count"] = summary["Count"].fillna(0).astype(int)

    summary_csv = os.path.join(experiment_root, f"phase_similarity_summary_{metric}_{matrix_type}_n{n}.csv")
    summary.to_csv(summary_csv, index=False, float_format="%.8f")
    print(f"Phase summary saved to: {summary_csv}")
    return summary, summary_csv


def plot_phase_similarity_lines(experiment_root, metric="deltacon", matrix_type="dual", n=10, sparsities=None):
    if sparsities is None:
        sparsities = [50, 70, 90, 98]
    summary, _ = build_phase_similarity_summary(
        experiment_root, metric=metric, matrix_type=matrix_type, n=n, sparsities=sparsities
    )
    plot_dir = os.path.join(experiment_root, "diagram")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for idx, sparsity in enumerate(sorted(sparsities)):
        sub = summary[summary["Sparsity"] == sparsity].copy()
        sub["Stage"] = pd.Categorical(sub["Stage"], categories=STAGE_ORDER, ordered=True)
        sub = sub.sort_values("Stage")
        x_labels = [STAGE_LABELS.get(stage, stage) for stage in sub["Stage"]]
        plt.plot(
            x_labels,
            sub["Avg_Similarity"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=SIMILARITY_COLORS[idx % len(SIMILARITY_COLORS)],
            label=f"Sparsity {sparsity}%",
        )

    if metric == "deltacon" and matrix_type == "dual":
        title = f"Average Dual DeltaCon Similarity by Stage (n={n})"
    else:
        title = f"Average {metric.capitalize()} Similarity by Stage (n={n})"
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Stage", fontsize=12)
    plt.ylabel("Average Similarity", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_path = os.path.join(plot_dir, f"phase_similarity_lines_{metric}_{matrix_type}_n{n}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Phase line plot saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "phases":
        parser = argparse.ArgumentParser()
        parser.add_argument("phases", nargs="?")
        parser.add_argument("--experiment-root", required=True, type=str)
        parser.add_argument("--sparsities", default="50,70,90,98", type=str)
        parser.add_argument("--metric", default="deltacon", choices=["deltacon", "jaccard"])
        parser.add_argument("--matrix-type", default="dual", type=str)
        parser.add_argument("--n", default=10, type=int)
        cli_args = parser.parse_args()
        try:
            plot_phase_similarity_lines(
                cli_args.experiment_root,
                metric=cli_args.metric,
                matrix_type=cli_args.matrix_type,
                n=cli_args.n,
                sparsities=_parse_sparsities(cli_args.sparsities),
            )
            print("Phase similarity plots generated successfully!")
            sys.exit(0)
        except (FileNotFoundError, ValueError) as e:
            print(f"Phase plotting failed: {e}")
            print("Please run analyze_similarity.py batch mode after placing adjacency data.")
            sys.exit(1)

    source = "set_mlp"
    matrix_type = "dual"

    args = sys.argv[1:]
    if args:
        if args[0] in VALID_SOURCES:
            source = args[0]
            args = args[1:]
        if args and args[0] in VALID_MATRIX_TYPES:
            matrix_type = args[0]
        elif args:
            print(f"Usage: python plot_similarity.py [source] [matrix_type]")
            print(f"  source: {VALID_SOURCES} (default: set_mlp)")
            print(f"  matrix_type: {VALID_MATRIX_TYPES} (default: dual)")
            sys.exit(1)

    print(f"Plotting: source={source}, matrix_type={matrix_type}")

    if matrix_type == "all":
        for mt in ("binary", "weighted", "dual"):
            plot_deltacon_similarity_by_n(mt, source)
            plot_jaccard_similarity_by_n(mt, source)
    else:
        plot_deltacon_similarity_by_n(matrix_type, source)
        plot_jaccard_similarity_by_n(matrix_type, source)

    print("All plots generated successfully!")
