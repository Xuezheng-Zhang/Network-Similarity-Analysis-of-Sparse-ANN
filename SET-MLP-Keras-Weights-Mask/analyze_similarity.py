import json
import numpy as np
import os
import sys
import argparse
import pandas as pd
from scipy.sparse import load_npz
import time
from multiprocessing import Pool, cpu_count

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deltacon_path = os.path.join(parent_dir, 'similarity_metrics')
if deltacon_path not in sys.path:
    sys.path.insert(0, deltacon_path)

from deltaCon import DeltaCon, load_adjacency_from_npz
from jaccard import Jaccard as JaccardSimilarity


def _similarity_task(args):
    file1, file2, g, matrix_type, epoch1_num, epoch2_num = args
    sim = compute_similarity(file1, file2, g, matrix_type)
    return epoch1_num, epoch2_num, sim

def _jaccard_similarity_task(args):
    file1, file2, matrix_type, epoch1_num, epoch2_num = args
    sim = compute_jaccard_similarity(file1, file2, matrix_type)
    return epoch1_num, epoch2_num, sim


def _similarity_dual_task(args):
    if len(args) == 9:
        file1_pos, file2_pos, file1_neg, file2_neg, g, epoch1_num, epoch2_num, w_pos, w_neg = args
        sim = compute_similarity_dual(file1_pos, file2_pos, file1_neg, file2_neg, g, w_pos=w_pos, w_neg=w_neg)
    else:
        file1_pos, file2_pos, file1_neg, file2_neg, g, epoch1_num, epoch2_num = args
        sim = compute_similarity_dual(file1_pos, file2_pos, file1_neg, file2_neg, g)
    return epoch1_num, epoch2_num, sim


def _jaccard_similarity_dual_task(args):
    if len(args) == 8:
        file1_pos, file2_pos, file1_neg, file2_neg, epoch1_num, epoch2_num, w_pos, w_neg = args
        sim = compute_jaccard_similarity_dual(file1_pos, file2_pos, file1_neg, file2_neg, w_pos=w_pos, w_neg=w_neg)
    else:
        file1_pos, file2_pos, file1_neg, file2_neg, epoch1_num, epoch2_num = args
        sim = compute_jaccard_similarity_dual(file1_pos, file2_pos, file1_neg, file2_neg)
    return epoch1_num, epoch2_num, sim


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

        if matrix_type == 'weighted':
            A1.data = np.abs(A1.data)
            A2.data = np.abs(A2.data)


        if A1.shape[0] != A2.shape[0]:
            print(f"Matrices do not have different sizes: {A1.shape[0]} vs {A2.shape[0]}")
            return None

        sim = DeltaCon(A1, A2, g)
        return sim
    except Exception as e:
        print(f"Error computing similarity between {file1} and {file2}: {e}")
        return None


def compute_similarity_dual(file1_pos, file2_pos, file1_neg, file2_neg, g=5, w_pos=None, w_neg=None):
    sim_pos = compute_similarity(file1_pos, file2_pos, g=g, matrix_type='weighted')
    sim_neg = compute_similarity(file1_neg, file2_neg, g=g, matrix_type='weighted')
    if sim_pos is None or sim_neg is None:
        return None
    if w_pos is not None and w_neg is not None:
        return w_pos * sim_pos + w_neg * sim_neg
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


def compute_jaccard_similarity_dual(file1_pos, file2_pos, file1_neg, file2_neg, w_pos=None, w_neg=None):
    sim_pos = compute_jaccard_similarity(file1_pos, file2_pos, matrix_type='weighted')
    sim_neg = compute_jaccard_similarity(file1_neg, file2_neg, matrix_type='weighted')
    if sim_pos is None or sim_neg is None:
        return None
    if w_pos is not None and w_neg is not None:
        return w_pos * sim_pos + w_neg * sim_neg
    return 0.5 * sim_pos + 0.5 * sim_neg


def analyze_epoch_similarities(base_dir, output_file, g=5, n=1, matrix_type='binary', run_id=None):
    epoch_dirs = get_epoch_dirs(base_dir)

    if len(epoch_dirs) < 2:
        print(f"Error: Need at least 2 epochs, found {len(epoch_dirs)}")
        return

    print(f"Found {len(epoch_dirs)} epochs" + (f" (run {run_id})" if run_id is not None else ""))
    print(f"Matrix type: {matrix_type}")
    print(f"DeltaCon parameter g: {g}")
    print(f"Step size n: {n}")
    print("-" * 70)

    results = []

    print("\nComputing similarities between epochs (after_training)")
    tasks = []
    if matrix_type == 'dual':
        sample_pos = os.path.join(base_dir, epoch_dirs[0], 'after_training_positive.npz')
        sample_neg = os.path.join(base_dir, epoch_dirs[0], 'after_training_negative.npz')
        if not os.path.isfile(sample_pos) or not os.path.isfile(sample_neg):
            print(f"  Dual mode requires after_training_positive.npz and after_training_negative.npz.")
            print(f"  Missing in e.g. {base_dir}/{epoch_dirs[0]}/")
            print(f"  Run: python convert_to_adjacency.py [static|set_mlp|rbm|all]  to generate them.")
            return
        results_dir = os.path.dirname(output_file)
        lookup = _load_weight_sign_lookup(results_dir)
        source = _infer_source_from_base_dir(base_dir)
        for i in range(0, len(epoch_dirs) - n, n):
            epoch1 = epoch_dirs[i]
            epoch2 = epoch_dirs[i + n]
            epoch1_num = int(epoch1.split('_')[1])
            epoch2_num = int(epoch2.split('_')[1])
            file1_pos = os.path.join(base_dir, epoch1, 'after_training_positive.npz')
            file2_pos = os.path.join(base_dir, epoch2, 'after_training_positive.npz')
            file1_neg = os.path.join(base_dir, epoch1, 'after_training_negative.npz')
            file2_neg = os.path.join(base_dir, epoch2, 'after_training_negative.npz')
            w_pos, w_neg = _dual_weights_for_pair(lookup, source, run_id, epoch1_num, epoch2_num)
            tasks.append((file1_pos, file2_pos, file1_neg, file2_neg, g, epoch1_num, epoch2_num, w_pos, w_neg))
    else:
        for i in range(0, len(epoch_dirs) - n, n):
            epoch1 = epoch_dirs[i]
            epoch2 = epoch_dirs[i + n]
            epoch1_num = int(epoch1.split('_')[1])
            epoch2_num = int(epoch2.split('_')[1])
            file1 = os.path.join(base_dir, epoch1, f'after_training_{matrix_type}.npz')
            file2 = os.path.join(base_dir, epoch2, f'after_training_{matrix_type}.npz')
            tasks.append((file1, file2, g, matrix_type, epoch1_num, epoch2_num))

    if not tasks:
        print("No epoch pairs to compare.")
        return

    num_workers = min(cpu_count(), len(tasks))
    print(f"Using {num_workers} parallel workers")

    if matrix_type == 'dual':
        with Pool(processes=num_workers) as pool:
            for epoch1_num, epoch2_num, sim in pool.imap_unordered(_similarity_dual_task, tasks):
                print(f"  Computing: Epoch {epoch1_num} -> Epoch {epoch2_num}", end=' ')
                if sim is not None:
                    row = {
                        'Type': 'Between_Epochs',
                        'Epoch1': epoch1_num,
                        'Epoch2': epoch2_num,
                        'Stage1': 'after_training',
                        'Stage2': 'after_training',
                        'Similarity': sim,
                        'Matrix_Type': matrix_type,
                        'n': n
                    }
                    if run_id is not None:
                        row['Run'] = run_id
                    results.append(row)
                    print(f"Similarity: {sim:.6f}")
                else:
                    print("Failed")
    else:
        with Pool(processes=num_workers) as pool:
            for epoch1_num, epoch2_num, sim in pool.imap_unordered(_similarity_task, tasks):
                print(f"  Computing: Epoch {epoch1_num} -> Epoch {epoch2_num}", end=' ')
                if sim is not None:
                    row = {
                        'Type': 'Between_Epochs',
                        'Epoch1': epoch1_num,
                        'Epoch2': epoch2_num,
                        'Stage1': 'after_training',
                        'Stage2': 'after_training',
                        'Similarity': sim,
                        'Matrix_Type': matrix_type,
                        'n': n
                    }
                    if run_id is not None:
                        row['Run'] = run_id
                    results.append(row)
                    print(f"Similarity: {sim:.6f}")
                else:
                    print("Failed")






























    if not results:
        return
    df = pd.DataFrame(results)
    column_order = ['Type', 'Epoch1', 'Epoch2', 'Stage1', 'Stage2', 'Similarity', 'Matrix_Type', 'n']
    if run_id is not None:
        column_order = ['Type', 'Run'] + [c for c in column_order if c != 'Type']
    df = df[column_order]
    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', index=False, float_format='%.8f', header=not file_exists)

def analyze_jaccard_similarities(base_dir, output_file, n=1, matrix_type='binary', run_id=None):
    epoch_dirs = get_epoch_dirs(base_dir)

    if len(epoch_dirs) < 2:
        print(f"Error: Need at least 2 epochs, found {len(epoch_dirs)}")
        return

    print(f"Found {len(epoch_dirs)} epochs" + (f" (run {run_id})" if run_id is not None else ""))
    print(f"Matrix type: {matrix_type}")
    print(f"Similarity method: Jaccard")
    print(f"Step size n: {n}")
    print("-" * 70)

    results = []

    print("\nComputing Jaccard similarities between epochs (after_training)")
    tasks = []
    if matrix_type == 'dual':
        sample_pos = os.path.join(base_dir, epoch_dirs[0], 'after_training_positive.npz')
        sample_neg = os.path.join(base_dir, epoch_dirs[0], 'after_training_negative.npz')
        if not os.path.isfile(sample_pos) or not os.path.isfile(sample_neg):
            print(f"  Dual mode requires after_training_positive.npz and after_training_negative.npz.")
            print(f"  Missing in e.g. {base_dir}/{epoch_dirs[0]}/")
            print(f"  Run: python convert_to_adjacency.py [static|set_mlp|rbm|all]  to generate them.")
            return
        results_dir = os.path.dirname(output_file)
        lookup = _load_weight_sign_lookup(results_dir)
        source = _infer_source_from_base_dir(base_dir)
        for i in range(0, len(epoch_dirs) - n, n):
            epoch1 = epoch_dirs[i]
            epoch2 = epoch_dirs[i + n]
            epoch1_num = int(epoch1.split('_')[1])
            epoch2_num = int(epoch2.split('_')[1])
            file1_pos = os.path.join(base_dir, epoch1, 'after_training_positive.npz')
            file2_pos = os.path.join(base_dir, epoch2, 'after_training_positive.npz')
            file1_neg = os.path.join(base_dir, epoch1, 'after_training_negative.npz')
            file2_neg = os.path.join(base_dir, epoch2, 'after_training_negative.npz')
            w_pos, w_neg = _dual_weights_for_pair(lookup, source, run_id, epoch1_num, epoch2_num)
            tasks.append((file1_pos, file2_pos, file1_neg, file2_neg, epoch1_num, epoch2_num, w_pos, w_neg))
    else:
        for i in range(0, len(epoch_dirs) - n, n):
            epoch1 = epoch_dirs[i]
            epoch2 = epoch_dirs[i + n]
            epoch1_num = int(epoch1.split('_')[1])
            epoch2_num = int(epoch2.split('_')[1])
            file1 = os.path.join(base_dir, epoch1, f'after_training_{matrix_type}.npz')
            file2 = os.path.join(base_dir, epoch2, f'after_training_{matrix_type}.npz')
            tasks.append((file1, file2, matrix_type, epoch1_num, epoch2_num))

    if not tasks:
        print("No epoch pairs to compare.")
        return

    num_workers = min(cpu_count(), len(tasks))
    print(f"Using {num_workers} parallel workers")

    if matrix_type == 'dual':
        with Pool(processes=num_workers) as pool:
            for epoch1_num, epoch2_num, sim in pool.imap_unordered(_jaccard_similarity_dual_task, tasks):
                print(f"  Computing: Epoch {epoch1_num} -> Epoch {epoch2_num}", end=' ')
                if sim is not None:
                    row = {
                        'Type': 'Between_Epochs',
                        'Epoch1': epoch1_num,
                        'Epoch2': epoch2_num,
                        'Stage1': 'after_training',
                        'Stage2': 'after_training',
                        'Similarity': sim,
                        'Matrix_Type': matrix_type,
                        'n': n
                    }
                    if run_id is not None:
                        row['Run'] = run_id
                    results.append(row)
                    print(f"Jaccard Similarity: {sim:.6f}")
                else:
                    print("Failed")
    else:
        with Pool(processes=num_workers) as pool:
            for epoch1_num, epoch2_num, sim in pool.imap_unordered(_jaccard_similarity_task, tasks):
                print(f"  Computing: Epoch {epoch1_num} -> Epoch {epoch2_num}", end=' ')
                if sim is not None:
                    row = {
                        'Type': 'Between_Epochs',
                        'Epoch1': epoch1_num,
                        'Epoch2': epoch2_num,
                        'Stage1': 'after_training',
                        'Stage2': 'after_training',
                        'Similarity': sim,
                        'Matrix_Type': matrix_type,
                        'n': n
                    }
                    if run_id is not None:
                        row['Run'] = run_id
                    results.append(row)
                    print(f"Jaccard Similarity: {sim:.6f}")
                else:
                    print("Failed")

    if not results:
        return

    df = pd.DataFrame(results)
    column_order = ['Type', 'Epoch1', 'Epoch2', 'Stage1', 'Stage2', 'Similarity', 'Matrix_Type', 'n']
    if run_id is not None:
        column_order = ['Type', 'Run'] + [c for c in column_order if c != 'Type']
    df = df[column_order]
    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', index=False, float_format='%.8f', header=not file_exists)

def get_run_dirs(adj_base):
    if not os.path.isdir(adj_base):
        return []
    out = []
    for d in os.listdir(adj_base):
        path = os.path.join(adj_base, d)
        if os.path.isdir(path) and d.startswith('run_'):
            try:
                i = int(d.replace('run_', ''))
                out.append((i, d))
            except ValueError:
                pass
    out.sort(key=lambda x: x[0])
    return [d for _, d in out]


def _infer_source_from_base_dir(base_dir):
    if 'adjacency_matrices_static' in base_dir:
        return 'static'
    if 'adjacency_matrices_rbm' in base_dir:
        return 'rbm'
    return 'set_mlp'


def _load_weight_sign_lookup(results_dir):
    path = os.path.join(results_dir, 'weight_sign_stats.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    lookup = {}
    for r in data:
        if r.get('layer') != 'all' or r.get('stage') != 'after_training':
            continue
        run = r.get('run')
        if run == '' or run is None:
            run_key = ''
        else:
            run_key = int(run) if isinstance(run, int) else run
        key = (r['source'], run_key, r['epoch'])
        pct_pos = float(r.get('pct_positive_of_nonz', 50))
        pct_neg = float(r.get('pct_negative_of_nonz', 50))
        lookup[key] = (pct_pos / 100.0, pct_neg / 100.0)
    return lookup if lookup else None


def _dual_weights_for_pair(lookup, source, run_id, epoch1_num, epoch2_num):
    if lookup is None:
        return 0.5, 0.5
    run_key = '' if run_id is None else run_id
    k1 = (source, run_key, epoch1_num)
    k2 = (source, run_key, epoch2_num)
    w1 = lookup.get(k1)
    w2 = lookup.get(k2)
    if w1 is None or w2 is None:
        return 0.5, 0.5
    w_pos = (w1[0] + w2[0]) / 2.0
    w_neg = (w1[1] + w2[1]) / 2.0
    return w_pos, w_neg


SOURCES = {
    "set_mlp": "SET-MLP-Keras-Weights-Mask/results/adjacency_matrices",
    "rbm": "SET-MLP-Keras-Weights-Mask/results/adjacency_matrices_rbm",
    "static": "SET-MLP-Keras-Weights-Mask/results/adjacency_matrices_static",
}


def analyze_source(source_name, adj_base, output_dir, g, n_values, matrix_type):
    if not os.path.exists(adj_base):
        print(f"Skipping {source_name}: {adj_base} not found")
        return

    print(f"# Analyzing source: {source_name}")
    print(f"# adj_base: {adj_base}")

    run_dirs = get_run_dirs(adj_base)
    if not run_dirs:
        run_dirs = [None]
        base_dirs = [adj_base]
        run_ids = [None]
    else:
        base_dirs = [os.path.join(adj_base, r) for r in run_dirs]
        run_ids = [int(r.replace('run_', '')) for r in run_dirs]

    deltacon_csv = os.path.join(output_dir, f'deltacon_similarity_{source_name}_{matrix_type}.csv')
    jaccard_csv = os.path.join(output_dir, f'jaccard_similarity_{source_name}_{matrix_type}.csv')

    if os.path.exists(deltacon_csv):
        os.remove(deltacon_csv)
    if os.path.exists(jaccard_csv):
        os.remove(jaccard_csv)

    for n in n_values:
        print(f"\nRunning analyses with step size n={n}")
        for idx, base_dir in enumerate(base_dirs):
            run_id = run_ids[idx]
            run_label = f"run_{run_id}" if run_id is not None else "single"
            print(f"\n{'='*60}\n>>> {source_name}/{run_label} (n={n})\n{'='*60}")

            start = time.time()
            analyze_epoch_similarities(base_dir, deltacon_csv, g=g, n=n, matrix_type=matrix_type, run_id=run_id)
            print(f"DeltaCon ({source_name}, {run_label}, n={n}): {time.time() - start:.2f}s")

            start = time.time()
            analyze_jaccard_similarities(base_dir, jaccard_csv, n=n, matrix_type=matrix_type, run_id=run_id)
            print(f"Jaccard ({source_name}, {run_label}, n={n}): {time.time() - start:.2f}s")

    print(f"\n[{source_name}] Results written to {deltacon_csv} and {jaccard_csv}")


def _parse_sparsities(sparsities_arg):
    return [int(x.strip()) for x in sparsities_arg.split(",") if x.strip()]


def _append_sparsity_column(input_csv, output_csv, sparsity):
    if not os.path.isfile(input_csv):
        print(f"Missing input CSV for sparsity={sparsity}: {input_csv}")
        return
    df = pd.read_csv(input_csv)
    if df.empty:
        return
    df["Sparsity"] = sparsity
    ordered = ["Sparsity"] + [c for c in df.columns if c != "Sparsity"]
    df = df[ordered]
    file_exists = os.path.exists(output_csv)
    df.to_csv(output_csv, mode="a", index=False, float_format="%.8f", header=not file_exists)


def analyze_experiments(experiment_root, sparsities, g=5, n=10, matrix_type="dual"):
    os.makedirs(experiment_root, exist_ok=True)
    n_values = [n]
    combined_dir = os.path.join(experiment_root, "similarity")
    os.makedirs(combined_dir, exist_ok=True)
    combined_delta = os.path.join(
        combined_dir, f"deltacon_similarity_set_mlp_sparsities_{matrix_type}_n{n}.csv"
    )
    combined_jaccard = os.path.join(
        combined_dir, f"jaccard_similarity_set_mlp_sparsities_{matrix_type}_n{n}.csv"
    )
    if os.path.exists(combined_delta):
        os.remove(combined_delta)
    if os.path.exists(combined_jaccard):
        os.remove(combined_jaccard)

    wrote_delta = False
    wrote_jaccard = False
    for sparsity in sparsities:
        exp_dir = os.path.join(experiment_root, f"set_mlp_s{sparsity}")
        adj_base = os.path.join(exp_dir, "adjacency_matrices")
        if not os.path.isdir(adj_base):
            print(f"Skipping sparsity={sparsity}: missing adjacency directory {adj_base}")
            continue
        similarity_dir = os.path.join(exp_dir, "similarity")
        os.makedirs(similarity_dir, exist_ok=True)
        source_name = f"set_mlp_s{sparsity}"
        print(f"\n{'#' * 60}\n# Batch experiment sparsity={sparsity}\n{'#' * 60}")
        analyze_source(source_name, adj_base, similarity_dir, g, n_values, matrix_type)

        per_delta = os.path.join(similarity_dir, f"deltacon_similarity_{source_name}_{matrix_type}.csv")
        per_jaccard = os.path.join(similarity_dir, f"jaccard_similarity_{source_name}_{matrix_type}.csv")
        if os.path.isfile(per_delta):
            wrote_delta = True
        if os.path.isfile(per_jaccard):
            wrote_jaccard = True
        _append_sparsity_column(per_delta, combined_delta, sparsity)
        _append_sparsity_column(per_jaccard, combined_jaccard, sparsity)

    print("\nBatch experiment analysis complete.")
    if wrote_delta and os.path.isfile(combined_delta):
        print(f"Combined DeltaCon CSV: {combined_delta}")
    else:
        print("Combined DeltaCon CSV not created (no valid per-sparsity results found).")
    if wrote_jaccard and os.path.isfile(combined_jaccard):
        print(f"Combined Jaccard CSV: {combined_jaccard}")
    else:
        print("Combined Jaccard CSV not created (no valid per-sparsity results found).")


def _is_experiments_layout(root_dir):
    if not os.path.isdir(root_dir):
        return False
    for d in os.listdir(root_dir):
        full = os.path.join(root_dir, d)
        if os.path.isdir(full) and d.startswith("set_mlp_s"):
            return True
    return False


def _is_adjacency_layout(root_dir):
    if not os.path.isdir(root_dir):
        return False
    names = os.listdir(root_dir)
    return any(name.startswith("run_") or name.startswith("epoch_") for name in names)


def _infer_source_name_from_path(path):
    p = path.lower()
    if "adjacency_matrices_rbm" in p:
        return "rbm"
    if "adjacency_matrices_static" in p:
        return "static"
    return "set_mlp"


def analyze_adjacency_root(adj_root, g=5, n=10, matrix_type="dual"):
    output_dir = "SET-MLP-Keras-Weights-Mask/results/similarity"
    os.makedirs(output_dir, exist_ok=True)
    source_name = _infer_source_name_from_path(adj_root)
    print(
        f"Adjacency-root mode: root={adj_root}, source={source_name}, "
        f"g={g}, n={n}, matrix_type={matrix_type}"
    )
    analyze_source(source_name, adj_root, output_dir, g, [n], matrix_type)


def main():
    args = sys.argv[1:]
    if any(arg.startswith("--") for arg in args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment-root", type=str, default=None)
        parser.add_argument("--sparsities", type=str, default="50,70,90,98")
        parser.add_argument("--g", type=int, default=5)
        parser.add_argument("--n", type=int, default=10)
        parser.add_argument("--matrix-type", type=str, default="dual")
        parsed = parser.parse_args(args)
        if parsed.experiment_root:
            root_dir = parsed.experiment_root
            if _is_experiments_layout(root_dir):
                sparsities = _parse_sparsities(parsed.sparsities)
                print(
                    f"Batch mode: experiment_root={root_dir}, "
                    f"sparsities={sparsities}, g={parsed.g}, n={parsed.n}, matrix_type={parsed.matrix_type}"
                )
                analyze_experiments(
                    root_dir,
                    sparsities,
                    g=parsed.g,
                    n=parsed.n,
                    matrix_type=parsed.matrix_type,
                )
                print("\nAnalysis complete!")
                return
            if _is_adjacency_layout(root_dir):
                analyze_adjacency_root(
                    root_dir,
                    g=parsed.g,
                    n=parsed.n,
                    matrix_type=parsed.matrix_type,
                )
                print("\nAnalysis complete!")
                return
            return
        return

    output_dir = "SET-MLP-Keras-Weights-Mask/results/similarity"
    os.makedirs(output_dir, exist_ok=True)
    source = "set_mlp"
    g = 5
    n_values = [3, 5, 10, 20]
    matrix_type = "dual"
    if args:
        if args[0] in SOURCES or args[0] == "all":
            source = args[0]
            args = args[1:]
        if args:
            g = int(args[0])
        if len(args) > 1:
            n_values = [int(args[1])]
        if len(args) > 2:
            matrix_type = args[2]

    print(f"Source: {source}, g={g}, n_values={n_values}, matrix_type={matrix_type}")
    if source == "all":
        for name, adj_base in SOURCES.items():
            analyze_source(name, adj_base, output_dir, g, n_values, matrix_type)
    elif source in SOURCES:
        analyze_source(source, SOURCES[source], output_dir, g, n_values, matrix_type)
    else:
        print(f"Unknown source: '{source}'. Available: {list(SOURCES.keys())} or 'all'")
        print("Usage: python analyze_similarity.py [source] [g] [n] [matrix_type]")
        print("  source: set_mlp, rbm, static, or all (default: set_mlp)")
        return

    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
