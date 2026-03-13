import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np

RESULTS_DIR = "SET-MLP-Keras-Weights-Mask/results"
GRAPH_DIR = os.path.join(RESULTS_DIR, "diagram")

def load_val_accuracy_data():
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
    """Group by (Epoch1, Epoch2, n), average Similarity over n runs."""
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
    """Add val accuracy as green line on right y-axis; left y-axis blue (similarity), right y-axis green (accuracy)."""
    if not epoch_to_acc:
        return
    ax = plt.gca()
    ax.set_ylabel("Similarity Score", fontsize=12, color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.spines["left"].set_color("blue")
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
    """Set x-axis tick labels to 30 degrees counterclockwise on the main axes."""
    fig = plt.gcf()
    if not fig.axes:
        return
    main_ax = fig.axes[0]
    for label in main_ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def plot_deltacon_similarity_by_n(matrix_type="binary"):
    """Plot DeltaCon similarity. matrix_type: 'binary', 'weighted', or 'dual' (uses _dual.csv)."""
    if matrix_type == "dual":
        plot_deltacon_similarity_dual_by_n()
        return
    similarity_file = os.path.join(RESULTS_DIR, f"deltacon_similarity_{matrix_type}.csv")
    if not os.path.isfile(similarity_file):
        print(f"No {similarity_file}; skip DeltaCon ({matrix_type}) plot.")
        return
    df = pd.read_csv(similarity_file)
    between_epochs = aggregate_similarity_mean(df)
    if between_epochs.empty:
        print(f"No Between_Epochs data for DeltaCon ({matrix_type}) plot; skip.")
        return

    epoch_to_acc = load_val_accuracy_data()
    unique_n_values = sorted(between_epochs["n"].unique())
    output_dir = GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    mean_over_runs = "Run" in df.columns
    name_suffix = "" if matrix_type == "binary" else f"_{matrix_type}"

    for n_val in unique_n_values:
        data_n = between_epochs[between_epochs["n"] == n_val].copy()
        data_n = data_n.sort_values("Epoch1")

        epochs = data_n["Epoch1"].values
        similarities = data_n["Similarity"].values
        epoch2_values = data_n["Epoch2"].values

        x_labels = [f"{int(e1)}" for e1 in epochs]
        plt.figure(figsize=(12, 8))
        ax_left = plt.gca()
        ax_left.plot(epochs, similarities, marker="o", linestyle="-", linewidth=2, markersize=8,
                    label=f"n={n_val}", color="blue")
        add_accuracy_line(epoch_to_acc)
        ax_left = plt.gcf().axes[0]
        ax_left.set_title(f"DeltaCon Similarity (step n={n_val})", fontsize=14, fontweight="bold", color="black")
        ax_left.set_xlabel('Epoch t to t+n', fontsize=12, color="black")
        ax_left.set_ylabel("Similarity Score", fontsize=12, color="blue")
        ax_left.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_left.set_xticks(epochs)
        ax_left.set_xticklabels(x_labels, rotation=30)
        ax_left.tick_params(axis="x", labelcolor="black")
        plt.tight_layout()
        _set_xlabel_rotation_30()
        output_path = os.path.join(output_dir, f"deltacon_similarity{name_suffix}_n_{n_val}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot for n={n_val} saved to: {output_path}")
        plt.close()

def plot_deltacon_similarity_dual_by_n():
    """Plot DeltaCon similarity for dual-channel (positive + negative) results."""
    similarity_file = os.path.join(RESULTS_DIR, "deltacon_similarity_dual.csv")
    if not os.path.isfile(similarity_file):
        print("No deltacon_similarity_dual.csv; skip DeltaCon dual plot.")
        return
    df = pd.read_csv(similarity_file)
    between_epochs = aggregate_similarity_mean(df)
    if between_epochs.empty:
        print("No Between_Epochs data for DeltaCon dual plot; skip.")
        return
    epoch_to_acc = load_val_accuracy_data()
    unique_n_values = sorted(between_epochs["n"].unique())
    output_dir = GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    for n_val in unique_n_values:
        data_n = between_epochs[between_epochs["n"] == n_val].copy()
        data_n = data_n.sort_values("Epoch1")
        epochs = data_n["Epoch1"].values
        similarities = data_n["Similarity"].values
        epoch2_values = data_n["Epoch2"].values
        x_labels = [f"{int(e1)}" for e1 in epochs]
        plt.figure(figsize=(12, 8))
        ax_left = plt.gca()
        ax_left.plot(epochs, similarities, marker="o", linestyle="-", linewidth=2, markersize=8,
                    label=f"n={n_val}", color="blue")
        add_accuracy_line(epoch_to_acc)
        ax_left = plt.gcf().axes[0]
        ax_left.set_title(f"DeltaCon Similarity (step n={n_val})", fontsize=14, fontweight="bold", color="black")
        ax_left.set_xlabel('Epoch t to t+n', fontsize=12, color="black")
        ax_left.set_ylabel("Similarity Score", fontsize=12, color="blue")
        ax_left.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_left.set_xticks(epochs)
        ax_left.set_xticklabels(x_labels, rotation=30)
        ax_left.tick_params(axis="x", labelcolor="black")
        plt.tight_layout()
        _set_xlabel_rotation_30()
        output_path = os.path.join(output_dir, f"deltacon_similarity_dual_n_{n_val}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"DeltaCon dual plot for n={n_val} saved to: {output_path}")
        plt.close()

def plot_jaccard_similarity_by_n(matrix_type="binary"):
    """Plot Jaccard similarity. matrix_type: 'binary', 'weighted', or 'dual' (uses _dual.csv)."""
    if matrix_type == "dual":
        plot_jaccard_similarity_dual_by_n()
        return
    similarity_file = os.path.join(RESULTS_DIR, f"jaccard_similarity_{matrix_type}.csv")
    if not os.path.isfile(similarity_file):
        print(f"No {similarity_file}; skip Jaccard ({matrix_type}) plot.")
        return
    df = pd.read_csv(similarity_file)
    between_epochs = aggregate_similarity_mean(df)
    if between_epochs.empty:
        print(f"No Between_Epochs data for Jaccard ({matrix_type}) plot; skip.")
        return

    epoch_to_acc = load_val_accuracy_data()
    unique_n_values = sorted(between_epochs["n"].unique())
    output_dir = GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    name_suffix = "" if matrix_type == "binary" else f"_{matrix_type}"

    for n_val in unique_n_values:
        data_n = between_epochs[between_epochs["n"] == n_val].copy()
        data_n = data_n.sort_values("Epoch1")

        epochs = data_n["Epoch1"].values
        similarities = data_n["Similarity"].values
        epoch2_values = data_n["Epoch2"].values

        x_labels = [f"{int(e1)}" for e1 in epochs]
        plt.figure(figsize=(12, 8))
        ax_left = plt.gca()
        ax_left.plot(epochs, similarities, marker="o", linestyle="-", linewidth=2, markersize=8,
                    label=f"n={n_val}", color="blue")
        add_accuracy_line(epoch_to_acc)
        ax_left = plt.gcf().axes[0]
        ax_left.set_title(f"Jaccard Similarity (step n={n_val})", fontsize=14, fontweight="bold", color="black")
        ax_left.set_xlabel('Epoch t to t+n', fontsize=12, color="black")
        ax_left.set_ylabel("Similarity Score", fontsize=12, color="blue")
        ax_left.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_left.set_xticks(epochs)
        ax_left.set_xticklabels(x_labels, rotation=30)
        ax_left.tick_params(axis="x", labelcolor="black")
        plt.tight_layout()
        _set_xlabel_rotation_30()
        output_path = os.path.join(output_dir, f"jaccard_similarity{name_suffix}_n_{n_val}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Jaccard plot for n={n_val} saved to: {output_path}")
        plt.close()

def plot_jaccard_similarity_dual_by_n():
    """Plot Jaccard similarity for dual-channel (positive + negative) results."""
    similarity_file = os.path.join(RESULTS_DIR, "jaccard_similarity_dual.csv")
    if not os.path.isfile(similarity_file):
        print("No jaccard_similarity_dual.csv; skip Jaccard dual plot.")
        return
    df = pd.read_csv(similarity_file)
    between_epochs = aggregate_similarity_mean(df)
    if between_epochs.empty:
        print("No Between_Epochs data for Jaccard dual plot; skip.")
        return
    epoch_to_acc = load_val_accuracy_data()
    unique_n_values = sorted(between_epochs["n"].unique())
    output_dir = GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    for n_val in unique_n_values:
        data_n = between_epochs[between_epochs["n"] == n_val].copy()
        data_n = data_n.sort_values("Epoch1")
        epochs = data_n["Epoch1"].values
        similarities = data_n["Similarity"].values
        epoch2_values = data_n["Epoch2"].values
        x_labels = [f"{int(e1)}" for e1 in epochs]
        plt.figure(figsize=(12, 8))
        ax_left = plt.gca()
        ax_left.plot(epochs, similarities, marker="o", linestyle="-", linewidth=2, markersize=8,
                    label=f"n={n_val}", color="blue")
        add_accuracy_line(epoch_to_acc)
        ax_left = plt.gcf().axes[0]
        ax_left.set_title(f"Jaccard Similarity (step n={n_val})", fontsize=14, fontweight="bold", color="black")
        ax_left.set_xlabel('Epoch t to t+n', fontsize=12, color="black")
        ax_left.set_ylabel("Similarity Score", fontsize=12, color="blue")
        ax_left.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_left.set_xticks(epochs)
        ax_left.set_xticklabels(x_labels, rotation=30)
        ax_left.tick_params(axis="x", labelcolor="black")
        plt.tight_layout()
        _set_xlabel_rotation_30()
        output_path = os.path.join(output_dir, f"jaccard_similarity_dual_n_{n_val}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Jaccard dual plot for n={n_val} saved to: {output_path}")
        plt.close()

def plot_similarity_between_epochs():
    similarity_file = "SET-MLP-Keras-Weights-Mask/results/deltacon_similarity_binary.csv"
    df = pd.read_csv(similarity_file)
    between_epochs = df[df['Type'] == 'Between_Epochs'].copy()

    # sort by epoch
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

    # sort by epoch
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

if __name__ == '__main__':
    # Usage: python plot_similarity.py [binary|weighted|dual|all]. Default: all
    matrix_type = "dual"
    if len(sys.argv) > 1:
        t = sys.argv[1].strip().lower()
        if t in ("binary", "weighted", "dual", "all"):
            matrix_type = t
        else:
            print("Usage: python plot_similarity.py [binary|weighted|dual|all]")
            print("  binary  - plot deltacon/jaccard_similarity_binary.csv")
            print("  weighted - plot deltacon/jaccard_similarity_weighted.csv")
            print("  dual    - plot deltacon_similarity_dual.csv & jaccard_similarity_dual.csv")
            print("  all     - plot binary, weighted, dual (default)")

    if matrix_type == "all":
        for mt in ("binary", "weighted", "dual"):
            plot_deltacon_similarity_by_n(mt)
            plot_jaccard_similarity_by_n(mt)
    else:
        plot_deltacon_similarity_by_n(matrix_type)
        plot_jaccard_similarity_by_n(matrix_type)
    
    print("All plots generated successfully!")
