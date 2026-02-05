import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

RESULTS_DIR = "SET-MLP-Keras-Weights-Mask/results"

def load_val_accuracy_data():
    """Load val accuracy; prefer mean over 5 runs (training_metadata_run_*.json)."""
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
    """Group by (Epoch1, Epoch2, n), average Similarity over 5 runs. Return aggregated df."""
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

def plot_deltacon_similarity_by_n():
    similarity_file = os.path.join(RESULTS_DIR, "deltacon_similarity_binary.csv")
    df = pd.read_csv(similarity_file)
    between_epochs = aggregate_similarity_mean(df)
    if between_epochs.empty:
        print("No Between_Epochs data for DeltaCon plot; skip.")
        return

    epoch_to_acc = load_val_accuracy_data()
    unique_n_values = sorted(between_epochs["n"].unique())
    output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    mean_over_runs = "Run" in df.columns

    for n_val in unique_n_values:
        data_n = between_epochs[between_epochs["n"] == n_val].copy()
        data_n = data_n.sort_values("Epoch1")

        epochs = data_n["Epoch1"].values
        similarities = data_n["Similarity"].values
        epoch2_values = data_n["Epoch2"].values

        x_labels = [f"{int(e1)}-{int(e2)}" for e1, e2 in zip(epochs, epoch2_values)]
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, similarities, marker="o", linestyle="-", linewidth=2, markersize=8,
                 label=f"n={n_val}", color="blue")

        for _, (epoch1, epoch2, sim) in enumerate(zip(epochs, epoch2_values, similarities)):
            accuracies = get_val_accuracy_range(int(epoch1), int(epoch2), epoch_to_acc)
            if accuracies is None or len(accuracies) == 0:
                continue
            acc_text = f"Acc: {accuracies[0]:.2f}" if len(accuracies) == 1 else f"Acc: {accuracies[0]:.2f}, {accuracies[1]:.2f}"
            plt.annotate(
                acc_text,
                xy=(epoch1, sim),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5, lw=0.5),
            )

        title = f"DeltaCon Similarity (step n={n_val}, mean over 5 runs)" if mean_over_runs else f"DeltaCon Similarity (step n={n_val})"
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel('Epoch t to t+n', fontsize=12)
        plt.ylabel('Similarity Score', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xticks(epochs, x_labels, rotation=10)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"deltacon_similarity_n_{n_val}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot for n={n_val} saved to: {output_path}")
        plt.close()

def plot_jaccard_similarity_by_n():
    similarity_file = os.path.join(RESULTS_DIR, "jaccard_similarity_binary.csv")
    df = pd.read_csv(similarity_file)
    between_epochs = aggregate_similarity_mean(df)
    if between_epochs.empty:
        print("No Between_Epochs data for Jaccard plot; skip.")
        return

    epoch_to_acc = load_val_accuracy_data()
    unique_n_values = sorted(between_epochs["n"].unique())
    output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    mean_over_runs = "Run" in df.columns

    for n_val in unique_n_values:
        data_n = between_epochs[between_epochs["n"] == n_val].copy()
        data_n = data_n.sort_values("Epoch1")

        epochs = data_n["Epoch1"].values
        similarities = data_n["Similarity"].values
        epoch2_values = data_n["Epoch2"].values

        x_labels = [f"{int(e1)}-{int(e2)}" for e1, e2 in zip(epochs, epoch2_values)]
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, similarities, marker="o", linestyle="-", linewidth=2, markersize=8,
                 label=f"n={n_val}", color="blue")

        for _, (epoch1, epoch2, sim) in enumerate(zip(epochs, epoch2_values, similarities)):
            accuracies = get_val_accuracy_range(int(epoch1), int(epoch2), epoch_to_acc)
            if accuracies is None or len(accuracies) == 0:
                continue
            acc_text = f"Acc: {accuracies[0]:.2f}" if len(accuracies) == 1 else f"Acc: {accuracies[0]:.2f}, {accuracies[1]:.2f}"
            plt.annotate(
                acc_text,
                xy=(epoch1, sim),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5, lw=0.5),
            )

        title = f"Jaccard Similarity (step n={n_val}, mean over 5 runs)" if mean_over_runs else f"Jaccard Similarity (step n={n_val})"
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel('Epoch t to t+n', fontsize=12)
        plt.ylabel('Similarity Score', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xticks(epochs, x_labels, rotation=10)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"jaccard_similarity_n_{n_val}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Jaccard plot for n={n_val} saved to: {output_path}")
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
    plt.plot(epochs, similarities, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    
    plt.title('DeltaCon Similarity Evolution', fontsize=14)
    plt.xlabel('Epoch Transition (t to t+1)', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(epochs)
    
    output_path = "SET-MLP-Keras-Weights-Mask/results/similarity_between_epochs.png"
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
    plt.plot(epochs, similarities, marker='o', linestyle='-', color='r', linewidth=2, markersize=6)
    
    plt.title('DeltaCon Similarity Evolution', fontsize=14)
    plt.xlabel('Epoch Transition (t to t+n)', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(epochs)
    
    output_path = "SET-MLP-Keras-Weights-Mask/results/similarity_within_epochs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Within_Epochs plot saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    # Plot DeltaCon similarity
    plot_deltacon_similarity_by_n()
    
    # Plot Jaccard similarity
    plot_jaccard_similarity_by_n()
    
    # Plot Between_Epochs similarity (legacy)
    # plot_similarity_between_epochs()
    
    # Plot Within_Epoch similarity (legacy)
    # plot_similarity_within_epochs()
    
    print("All plots generated successfully!")
