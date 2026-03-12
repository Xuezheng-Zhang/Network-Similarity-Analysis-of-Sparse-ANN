import numpy as np
import os
import sys
import pandas as pd
from scipy.sparse import load_npz
import time
from multiprocessing import Pool, cpu_count

# Import DeltaCon functions
# Add parent directory to path to import deltaCon module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deltacon_path = os.path.join(parent_dir, 'similarity_metrics')
if deltacon_path not in sys.path:
    sys.path.insert(0, deltacon_path)

from deltaCon import DeltaCon, load_adjacency_from_npz
from jaccard import Jaccard as JaccardSimilarity


def _random_similarity_task(args):
    """
    Helper function for parallel similarity computation with random graph.
    """
    epoch_file, random_graph_file, g, matrix_type, epoch_num = args
    sim = compute_similarity(epoch_file, random_graph_file, g, matrix_type)
    return epoch_num, sim

def _random_jaccard_similarity_task(args):
    """
    Helper function for parallel Jaccard similarity computation with random graph.
    """
    epoch_file, random_graph_file, matrix_type, epoch_num = args
    sim = compute_jaccard_similarity(epoch_file, random_graph_file, matrix_type)
    return epoch_num, sim


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

def compute_jaccard_similarity(file1, file2, matrix_type='binary'):
    """
    Compute Jaccard similarity between two adjacency matrices
    
    Parameters:
        file1: Path to first adjacency matrix file
        file2: Path to second adjacency matrix file
        matrix_type: 'binary' or 'weighted' (for Jaccard, only edge sets matter)
    
    Returns:
        float: Jaccard similarity score (0-1)
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
        
        sim = JaccardSimilarity(A1, A2)
        return sim
    except Exception as e:
        print(f"Error computing Jaccard similarity between {file1} and {file2}: {e}")
        return None


def analyze_random_similarity(base_dir, random_graph_file, output_file, g=5, matrix_type='binary'):
    """
    Analyze similarities between each epoch and a random graph
    
    Parameters:
        base_dir: Base directory containing adjacency matrices
        random_graph_file: Path to the random graph file
        output_file: Output CSV file path
        g: Number of partitions for DeltaCon
        matrix_type: 'binary' or 'weighted'
    """
    epoch_dirs = get_epoch_dirs(base_dir)
    
    if len(epoch_dirs) == 0:
        print(f"Error: No epochs found in {base_dir}")
        return
    
    print(f"Found {len(epoch_dirs)} epochs")
    print(f"Random graph: {random_graph_file}")
    print(f"Matrix type: {matrix_type}")
    print(f"DeltaCon parameter g: {g}")
    print("-" * 70)
    
    results = []
    
    # Build tasks for similarities between epochs and random graph
    print("\nComputing similarities between epochs and random graph")
    tasks = []
    for epoch_dir in epoch_dirs:
        epoch_num = int(epoch_dir.split('_')[1])
        epoch_file = os.path.join(base_dir, epoch_dir, f'after_training_{matrix_type}.npz')
        
        if os.path.exists(epoch_file):
            tasks.append((epoch_file, random_graph_file, g, matrix_type, epoch_num))
        else:
            print(f"  Epoch {epoch_num}: File not found, skipping")
    
    if not tasks:
        print("No epochs to compare.")
        return
    
    num_workers = min(cpu_count(), len(tasks))
    print(f"Using {num_workers} parallel workers")
    
    with Pool(processes=num_workers) as pool:
        for epoch_num, sim in pool.imap_unordered(_random_similarity_task, tasks):
            print(f"  Computing: Epoch {epoch_num} vs Random graph", end=' ')
            if sim is not None:
                results.append({
                    'Type': 'Random_Comparison',
                    'Epoch1': epoch_num,
                    'Epoch2': -1,
                    'Stage1': 'after_training',
                    'Stage2': 'random',
                    'Similarity': sim,
                    'Matrix_Type': matrix_type,
                    'n': -1
                })
                print(f"Similarity: {sim:.6f}")
            else:
                print("Failed")
    
    # Save results to CSV (append mode)
    if results:
        df = pd.DataFrame(results)
        column_order = ['Type', 'Epoch1', 'Epoch2', 'Stage1', 'Stage2', 'Similarity', 'Matrix_Type', 'n']
        df = df[column_order]
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(output_file)
        df.to_csv(output_file, mode='a', index=False, float_format='%.8f', header=not file_exists)
        print(f"\nResults saved to: {output_file}")

def analyze_random_jaccard_similarity(base_dir, random_graph_file, output_file, matrix_type='binary'):
    """
    Analyze Jaccard similarities between each epoch and a random graph
    
    Parameters:
        base_dir: Base directory containing adjacency matrices
        random_graph_file: Path to the random graph file
        output_file: Output CSV file path
        matrix_type: 'binary' or 'weighted'
    """
    epoch_dirs = get_epoch_dirs(base_dir)
    
    if len(epoch_dirs) == 0:
        print(f"Error: No epochs found in {base_dir}")
        return
    
    print(f"Found {len(epoch_dirs)} epochs")
    print(f"Random graph: {random_graph_file}")
    print(f"Matrix type: {matrix_type}")
    print(f"Similarity method: Jaccard")
    print("-" * 70)
    
    results = []
    
    # Build tasks for Jaccard similarities between epochs and random graph
    print("\nComputing Jaccard similarities between epochs and random graph")
    tasks = []
    for epoch_dir in epoch_dirs:
        epoch_num = int(epoch_dir.split('_')[1])
        epoch_file = os.path.join(base_dir, epoch_dir, f'after_training_{matrix_type}.npz')
        
        if os.path.exists(epoch_file):
            tasks.append((epoch_file, random_graph_file, matrix_type, epoch_num))
        else:
            print(f"  Epoch {epoch_num}: File not found, skipping")
    
    if not tasks:
        print("No epochs to compare.")
        return
    
    num_workers = min(cpu_count(), len(tasks))
    print(f"Using {num_workers} parallel workers")
    
    with Pool(processes=num_workers) as pool:
        for epoch_num, sim in pool.imap_unordered(_random_jaccard_similarity_task, tasks):
            print(f"  Computing: Epoch {epoch_num} vs Random graph", end=' ')
            if sim is not None:
                results.append({
                    'Type': 'Random_Comparison',
                    'Epoch1': epoch_num,
                    'Epoch2': -1,
                    'Stage1': 'after_training',
                    'Stage2': 'random',
                    'Similarity': sim,
                    'Matrix_Type': matrix_type,
                    'n': -1
                })
                print(f"Jaccard Similarity: {sim:.6f}")
            else:
                print("Failed")
    
    # Save results to CSV (append mode)
    if results:
        df = pd.DataFrame(results)
        column_order = ['Type', 'Epoch1', 'Epoch2', 'Stage1', 'Stage2', 'Similarity', 'Matrix_Type', 'n']
        df = df[column_order]
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(output_file)
        df.to_csv(output_file, mode='a', index=False, float_format='%.8f', header=not file_exists)
        print(f"\nResults saved to: {output_file}")


def main():
    base_dir = "SET-MLP-Keras-Weights-Mask/results/adjacency_matrices"
    output_dir = "SET-MLP-Keras-Weights-Mask/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to random graph (12082 nodes)
    random_graph_file = "similarity_metrics/test_graphs/random_sparse_1690_sparsity_0.9300.npz"
    
    # Check if random graph exists
    if not os.path.exists(random_graph_file):
        print(f"Error: Random graph file not found: {random_graph_file}")
        print("Please run similarity_metrics/generate_test_graphs.py first to generate the random graph.")
        return
    
    # Get parameters (g and matrix_type)
    if sys.argv[1:]:
        g = int(sys.argv[1])
    else:
        g = 5
    
    if sys.argv[2:]:
        matrix_type = sys.argv[2]
    else:
        matrix_type = 'binary'
    
    # DeltaCon similarity
    output_file = os.path.join(output_dir, f'deltacon_similarity_{matrix_type}.csv')
    start_time = time.time()
    analyze_random_similarity(base_dir, random_graph_file, output_file, g=g, matrix_type=matrix_type)
    end_time = time.time()
    
    print(f"\nDeltaCon processing time: {end_time - start_time:.2f} seconds")
    
    # Jaccard similarity
    jaccard_output_file = os.path.join(output_dir, f'jaccard_similarity_{matrix_type}.csv')
    start_time = time.time()
    analyze_random_jaccard_similarity(base_dir, random_graph_file, jaccard_output_file, matrix_type=matrix_type)
    end_time = time.time()
    
    print(f"\nJaccard processing time: {end_time - start_time:.2f} seconds")
    print("Analysis complete!")


if __name__ == '__main__':
    main()
