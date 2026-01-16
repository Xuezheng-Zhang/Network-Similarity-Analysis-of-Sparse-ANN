import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_similarity_between_epochs():
    similarity_file = "SET-MLP-Keras-Weights-Mask/results/deltacon_similarity_binary.csv"
    df = pd.read_csv(similarity_file)
    between_epochs = df[df['Type'] == 'Between_Epochs'].copy()
    
    
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
    
    epochs = within_epochs['Epoch1'].values
    similarities = within_epochs['Similarity'].values
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, similarities, marker='o', linestyle='-', color='r', linewidth=2, markersize=6)
    
    plt.title('DeltaCon Similarity Evolution', fontsize=14)
    plt.xlabel('Epoch Transition (t to t+1)', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(epochs)
    
    output_path = "SET-MLP-Keras-Weights-Mask/results/similarity_within_epochs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Within_Epochs plot saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    
    # Plot Between_Epochs similarity
    plot_similarity_between_epochs()
    
    # Plot Within_Epoch similarity
    plot_similarity_within_epochs()
    print("All plots generated successfully!")