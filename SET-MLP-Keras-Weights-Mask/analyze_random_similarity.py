import numpy as np
import os
import sys
import pandas as pd
from scipy.sparse import load_npz, csr_matrix, coo_matrix, save_npz
import time
from multiprocessing import Pool, cpu_count

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deltacon_path = os.path.join(parent_dir, 'similarity_metrics')
if deltacon_path not in sys.path:
    sys.path.insert(0, deltacon_path)

from deltaCon import DeltaCon, load_adjacency_from_npz
from jaccard import Jaccard as JaccardSimilarity


def _random_similarity_task(args):
    epoch_file, random_graph_file, g, matrix_type, epoch_num = args
    sim = compute_similarity(epoch_file, random_graph_file, g, matrix_type)
    return epoch_num, sim


def _random_similarity_dual_task(args):
    epoch_pos, random_pos, epoch_neg, random_neg, g, epoch_num = args
    sim = compute_similarity_dual(epoch_pos, random_pos, epoch_neg, random_neg, g)
    return epoch_num, sim

def _random_jaccard_similarity_task(args):
    epoch_file, random_graph_file, matrix_type, epoch_num = args
    sim = compute_jaccard_similarity(epoch_file, random_graph_file, matrix_type)
    return epoch_num, sim


def get_epoch_dirs(base_dir):
    epoch_dirs = []
    for d in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('epoch_'):
            epoch_dirs.append(d)

    epoch_dirs.sort(key=lambda x: int(x.split('_')[1]))
    return epoch_dirs


def compute_similarity(file1, file2, g=5, matrix_type='binary'):
    if not os.path.exists(file1):
        return None
    if not os.path.exists(file2):
        return None

    try:
        A1 = load_adjacency_from_npz(file1, make_undirected=False, remove_self_loops=True)
        A2 = load_adjacency_from_npz(file2, make_undirected=False, remove_self_loops=True)

        if A1.shape[0] != A2.shape[0]:
            print(f"Matrices do not have different sizes: {A1.shape[0]} vs {A2.shape[0]}")
            return None

        if matrix_type == 'binary':
            if A1.nnz > 0:
                A1.data[:] = 1.0
            if A2.nnz > 0:
                A2.data[:] = 1.0

        if matrix_type == 'weighted':
            if A1.nnz > 0:
                A1.data = np.abs(A1.data)
            if A2.nnz > 0:
                A2.data = np.abs(A2.data)

        sim = DeltaCon(A1, A2, g)
        return sim
    except Exception as e:
        print(f"Error computing similarity between {file1} and {file2}: {e}")
        return None


def compute_similarity_dual(file1_pos, file2_pos, file1_neg, file2_neg, g=5):
    sim_pos = compute_similarity(file1_pos, file2_pos, g=g, matrix_type='weighted')
    sim_neg = compute_similarity(file1_neg, file2_neg, g=g, matrix_type='weighted')
    if sim_pos is None or sim_neg is None:
        return None
    return 0.5 * sim_pos + 0.5 * sim_neg

def compute_jaccard_similarity(file1, file2, matrix_type='binary'):
    if not os.path.exists(file1):
        return None
    if not os.path.exists(file2):
        return None

    try:
        A1 = load_adjacency_from_npz(file1, make_undirected=False, remove_self_loops=True)
        A2 = load_adjacency_from_npz(file2, make_undirected=False, remove_self_loops=True)

        if A1.shape[0] != A2.shape[0]:
            print(f"Matrices do not have different sizes: {A1.shape[0]} vs {A2.shape[0]}")
            return None

        sim = JaccardSimilarity(A1, A2)
        return sim
    except Exception as e:
        print(f"Error computing Jaccard similarity between {file1} and {file2}: {e}")
        return None


def _ensure_random_dual_graph(random_graph_file):
    base, _ = os.path.splitext(random_graph_file)
    random_pos_file = base + "_positive.npz"
    random_neg_file = base + "_negative.npz"

    if os.path.exists(random_pos_file) and os.path.exists(random_neg_file):
        return random_pos_file, random_neg_file

    A = load_adjacency_from_npz(random_graph_file, make_undirected=False, remove_self_loops=True)
    A_coo = coo_matrix(A)
    data = A_coo.data
    rows = A_coo.row
    cols = A_coo.col

    pos_mask = data > 0
    neg_mask = data < 0

    if not np.any(pos_mask) and not np.any(neg_mask):
        pos_mask = np.ones_like(data, dtype=bool)

    A_pos = csr_matrix((data[pos_mask], (rows[pos_mask], cols[pos_mask])), shape=A_coo.shape)
    A_neg = csr_matrix((data[neg_mask], (rows[neg_mask], cols[neg_mask])), shape=A_coo.shape)

    save_npz(random_pos_file, A_pos)
    save_npz(random_neg_file, A_neg)

    print(f"Random graph split into:\n  {random_pos_file}\n  {random_neg_file}")
    return random_pos_file, random_neg_file


def analyze_random_similarity(base_dir, random_graph_file, output_file, g=5, matrix_type='binary'):
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

    print("\nComputing similarities between epochs and random graph")
    tasks = []
    use_dual = (matrix_type == 'dual')
    random_pos_file = None
    random_neg_file = None

    if use_dual:
        random_pos_file, random_neg_file = _ensure_random_dual_graph(random_graph_file)

    for epoch_dir in epoch_dirs:
        epoch_num = int(epoch_dir.split('_')[1])
        if use_dual:
            epoch_pos = os.path.join(base_dir, epoch_dir, 'after_training_positive.npz')
            epoch_neg = os.path.join(base_dir, epoch_dir, 'after_training_negative.npz')
            if os.path.exists(epoch_pos) and os.path.exists(epoch_neg):
                tasks.append((epoch_pos, random_pos_file, epoch_neg, random_neg_file, g, epoch_num))
            else:
                print(f"  Epoch {epoch_num}: positive/negative files not found, skipping")
        else:
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

    worker_fn = _random_similarity_dual_task if use_dual else _random_similarity_task

    with Pool(processes=num_workers) as pool:
        for epoch_num, sim in pool.imap_unordered(worker_fn, tasks):
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

    if results:
        df = pd.DataFrame(results)
        column_order = ['Type', 'Epoch1', 'Epoch2', 'Stage1', 'Stage2', 'Similarity', 'Matrix_Type', 'n']
        df = df[column_order]
        file_exists = os.path.exists(output_file)
        df.to_csv(output_file, mode='a', index=False, float_format='%.8f', header=not file_exists)
        print(f"\nResults saved to: {output_file}")

def analyze_random_jaccard_similarity(base_dir, random_graph_file, output_file, matrix_type='binary'):
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

    if results:
        df = pd.DataFrame(results)
        column_order = ['Type', 'Epoch1', 'Epoch2', 'Stage1', 'Stage2', 'Similarity', 'Matrix_Type', 'n']
        df = df[column_order]
        file_exists = os.path.exists(output_file)
        df.to_csv(output_file, mode='a', index=False, float_format='%.8f', header=not file_exists)
        print(f"\nResults saved to: {output_file}")


def main():
    adj_base = "SET-MLP-Keras-Weights-Mask/results/adjacency_matrices"
    output_dir = "SET-MLP-Keras-Weights-Mask/results"
    os.makedirs(output_dir, exist_ok=True)

    random_graph_file = "similarity_metrics/test_graphs/random_sparse_1690_sparsity_0.9000.npz"
    if not os.path.exists(random_graph_file):
        print(f"Error: Random graph file not found: {random_graph_file}")
        print("Please run similarity_metrics/generate_test_graphs.py first to generate the random graph.")
        return

    if sys.argv[1:]:
        g = int(sys.argv[1])
    else:
        g = 5

    if sys.argv[2:]:
        matrix_type = sys.argv[2]
    else:
        matrix_type = 'weighted'

    if not os.path.isdir(adj_base):
        print(f"Error: adjacency base dir not found: {adj_base}")
        return

    run_dirs = []
    for d in os.listdir(adj_base):
        path = os.path.join(adj_base, d)
        if os.path.isdir(path) and d.startswith('run_'):
            run_dirs.append(d)

    if not run_dirs:
        base_dirs = [adj_base]
        run_labels = ["single"]
    else:
        run_dirs.sort(key=lambda x: int(x.replace('run_', '')))
        base_dirs = [os.path.join(adj_base, d) for d in run_dirs]
        run_labels = run_dirs

    output_file = os.path.join(output_dir, f'deltacon_similarity_{matrix_type}.csv')
    for base_dir, run_label in zip(base_dirs, run_labels):
        print(f"\n=== Random DeltaCon for {run_label} ===")
        start_time = time.time()
        analyze_random_similarity(base_dir, random_graph_file, output_file, g=g, matrix_type=matrix_type)
        end_time = time.time()
        print(f"DeltaCon processing time ({run_label}): {end_time - start_time:.2f} seconds")

    if matrix_type != 'dual':
        jaccard_output_file = os.path.join(output_dir, f'jaccard_similarity_{matrix_type}.csv')
        for base_dir, run_label in zip(base_dirs, run_labels):
            print(f"\n=== Random Jaccard for {run_label} ===")
            start_time = time.time()
            analyze_random_jaccard_similarity(base_dir, random_graph_file, jaccard_output_file, matrix_type=matrix_type)
            end_time = time.time()
            print(f"Jaccard processing time ({run_label}): {end_time - start_time:.2f} seconds")
    print("Analysis complete!")


if __name__ == '__main__':
    main()
