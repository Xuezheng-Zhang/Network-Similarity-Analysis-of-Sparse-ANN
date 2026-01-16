import numpy as np
import os
import sys
import pandas as pd
from scipy.sparse import load_npz
import time

# Import DeltaCon functions
# Add parent directory to path to import deltaCon module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deltacon_path = os.path.join(parent_dir, 'deltacon')
if deltacon_path not in sys.path:
    sys.path.insert(0, deltacon_path)

from deltaCon import DeltaCon, load_adjacency_from_npz

def get_epoch_dirs(base_dir):
    """
    Get all epoch directories sorted by epoch number
    """
    epoch_dirs = []
    for d in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('epoch_'):
            epoch_dirs.append(d)
    
    # Sort by epoch number
    epoch_dirs.sort(key=lambda x: int(x.split('_')[1]))
    return epoch_dirs

def compute_similarity(file1, file2, g=5, matrix_type='binary'):
    """
    Compute DeltaCon similarity between two adjacency matrices
    
    Parameters:
        file1: Path to first adjacency matrix file
        file2: Path to second adjacency matrix file
        g: Number of partitions for DeltaCon
        matrix_type: 'binary' or 'weighted'
    
    Returns:
        float: Similarity score (0-1)
    """
    if not os.path.exists(file1):
        return None
    if not os.path.exists(file2):
        return None
    
    try:
        A1 = load_adjacency_from_npz(file1, make_undirected=False, remove_self_loops=True)
        A2 = load_adjacency_from_npz(file2, make_undirected=False, remove_self_loops=True)
        
        # Check if matrices have the same size
        if A1.shape[0] != A2.shape[0]:
            print(f"Matrices do not have different sizes: {A1.shape[0]} vs {A2.shape[0]}")
            return None
        
        sim = DeltaCon(A1, A2, g)
        return sim
    except Exception as e:
        print(f"Error computing similarity between {file1} and {file2}: {e}")
        return None

def analyze_epoch_similarities(base_dir, output_file, g=5, n=1, matrix_type='binary'):
    """
    Analyze similarities between consecutive epochs and within each epoch
    
    Parameters:
        base_dir: Base directory containing adjacency matrices
        output_file: Output file path to save results
        g: Number of partitions for DeltaCon
        matrix_type: 'binary' or 'weighted'
    """
    epoch_dirs = get_epoch_dirs(base_dir)
    
    if len(epoch_dirs) < 2:
        print(f"Error: Need at least 2 epochs, found {len(epoch_dirs)}")
        return
    
    print(f"Found {len(epoch_dirs)} epochs")
    print(f"Matrix type: {matrix_type}")
    print(f"DeltaCon parameter g: {g}")
    print(f"Number of epochs to compare: {n}")
    print("-" * 70)
    
    results = []
    
    # Compute similarities between epoch and next n epoch (after_training)
    print("\nComputing similarities between epochs (after_training)")
    for i in range(len(epoch_dirs) - n):
        epoch1 = epoch_dirs[i]
        epoch2 = epoch_dirs[i + n]
        epoch1_num = int(epoch1.split('_')[1])
        epoch2_num = int(epoch2.split('_')[1])
        
        file1 = os.path.join(base_dir, epoch1, f'after_training_{matrix_type}.npz')
        file2 = os.path.join(base_dir, epoch2, f'after_training_{matrix_type}.npz')
        
        print(f"  Computing: Epoch {epoch1_num} -> Epoch {epoch2_num}", end=' ')
        sim = compute_similarity(file1, file2, g, matrix_type)
        
        if sim is not None:
            results.append({
                'Type': 'Between_Epochs',
                'Epoch1': epoch1_num,
                'Epoch2': epoch2_num,
                'Stage1': 'after_training',
                'Stage2': 'after_training',
                'Similarity': sim,
                'Matrix_Type': matrix_type
            })
            print(f"Similarity: {sim:.6f}")
        else:
            print("Failed")
    
    # # 2. Compute similarities within each epoch (after_training vs after_pruning)
    # print("\n2. Computing similarities within epochs (after_training vs after_pruning) ")
    # for epoch_dir in epoch_dirs:
    #     epoch_num = int(epoch_dir.split('_')[1])
        
    #     file1 = os.path.join(base_dir, epoch_dir, f'after_training_{matrix_type}.npz')
    #     file2 = os.path.join(base_dir, epoch_dir, f'after_pruning_{matrix_type}.npz')
        
    #     if not os.path.exists(file2):
    #         print(f"  Epoch {epoch_num}: after_pruning not found, skipping ")
    #         continue
        
    #     print(f"  Computing: Epoch {epoch_num} (training vs pruning)", end=' ')
    #     sim = compute_similarity(file1, file2, g, matrix_type)
        
    #     if sim is not None:
    #         results.append({
    #             'Type': 'Within_Epoch',
    #             'Epoch1': epoch_num,
    #             'Epoch2': epoch_num,
    #             'Stage1': 'after_training',
    #             'Stage2': 'after_pruning',
    #             'Similarity': sim,
    #             'Matrix_Type': matrix_type
    #         })
    #         print(f"Similarity: {sim:.6f}")
    #     else:
    #         print("Failed")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        column_order = ['Type', 'Epoch1', 'Epoch2', 'Stage1', 'Stage2', 'Similarity', 'Matrix_Type']
        df = df[column_order]
        df.to_csv(output_file, index=False, float_format='%.8f')
        print(f"Results saved to: {output_file}")
    

def main():
    base_dir = "SET-MLP-Keras-Weights-Mask/results/adjacency_matrices"
    output_dir = "SET-MLP-Keras-Weights-Mask/results"
    os.makedirs(output_dir, exist_ok=True)
    
    
    # get parameters (g and matrix_type)
    if sys.argv[1:]:
        g = int(sys.argv[1])
    else:
        g = 5

    if sys.argv[2:]:
        n = int(sys.argv[2])
    else:
        n = 1

    if sys.argv[3:]:
        matrix_type = sys.argv[3]
    else:
        matrix_type = 'binary'
        
    output_file = os.path.join(output_dir, f'deltacon_similarity_{matrix_type}.csv')
    start_time = time.time()
    analyze_epoch_similarities(base_dir, output_file, g=g, n=n, matrix_type=matrix_type)
    end_time = time.time()
    
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
    print("Analysis complete!")

if __name__ == '__main__':
    main()
