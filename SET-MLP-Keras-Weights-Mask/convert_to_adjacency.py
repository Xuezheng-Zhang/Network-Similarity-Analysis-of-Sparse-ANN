import numpy as np
import os
import sys
import argparse
from scipy.sparse import csr_matrix, save_npz, load_npz



SOURCE = "all"

def load_weights_from_directory(stage_dir):
    weights = {}
    available_layers = []
    for layer in [1, 2, 3, 4]:
        weight_file_npz = os.path.join(stage_dir, f"weight_layer_{layer}.npz")

        if os.path.exists(weight_file_npz):
            sparse_weight = load_npz(weight_file_npz)
            weights[f'layer_{layer}'] = sparse_weight.toarray()
            available_layers.append(layer)
    return weights, available_layers

def calculate_layer_sizes(weights_dict, available_layers):
    layer_names = sorted([f'layer_{i}' for i in available_layers])

    nodes_per_layer = {}
    offsets = {}

    if len(layer_names) == 0:
        return {}, {}, 0, []

    W1 = weights_dict[layer_names[0]]
    input_size = W1.shape[0]
    layer1_size = W1.shape[1]

    nodes_per_layer['input'] = input_size
    offsets['input'] = 0
    current_offset = input_size

    if len(layer_names) == 1:
        nodes_per_layer['output'] = layer1_size
        offsets['output'] = current_offset
    else:
        nodes_per_layer['layer1'] = layer1_size
        offsets['layer1'] = current_offset
        current_offset += layer1_size

        for idx, layer_name in enumerate(layer_names[1:-1], start=2):
            W = weights_dict[layer_name]
            layer_size = W.shape[1]
            nodes_per_layer[f'layer{idx}'] = layer_size
            offsets[f'layer{idx}'] = current_offset
            current_offset += layer_size

        last_layer_name = layer_names[-1]
        last_W = weights_dict[last_layer_name]
        output_size = last_W.shape[1]
        nodes_per_layer['output'] = output_size
        offsets['output'] = current_offset

    total_nodes = sum(nodes_per_layer.values())

    return nodes_per_layer, offsets, total_nodes, layer_names

def build_adjacency_matrix(weights_dict, available_layers, binary=False):
    _, offsets, total_nodes, layer_names = calculate_layer_sizes(weights_dict, available_layers)

    rows = []
    cols = []
    data = []

    for idx, layer_name in enumerate(layer_names):
        W = weights_dict[layer_name]

        if len(layer_names) == 1:
            src_name = 'input'
            dst_name = 'output'
        elif idx == 0:
            src_name = 'input'
            dst_name = 'layer1'
        elif idx == len(layer_names) - 1:
            src_name = f'layer{idx}'
            dst_name = 'output'
        else:
            src_name = f'layer{idx}'
            dst_name = f'layer{idx+1}'

        if binary:
            i, j = np.nonzero(W)
            rows.extend(offsets[src_name] + i)
            cols.extend(offsets[dst_name] + j)
            data.extend(np.ones(len(i), dtype=np.float32))
        else:
            i, j = np.nonzero(W)
            rows.extend(offsets[src_name] + i)
            cols.extend(offsets[dst_name] + j)
            data.extend(W[i, j].astype(np.float32))

    adj_matrix = csr_matrix((data, (rows, cols)),
                            shape=(total_nodes, total_nodes),
                            dtype=np.float32)
    return adj_matrix


def build_adjacency_matrix_dual(weights_dict, available_layers):
    _, offsets, total_nodes, layer_names = calculate_layer_sizes(
        weights_dict, available_layers
    )

    rows_pos, cols_pos, data_pos = [], [], []
    rows_neg, cols_neg, data_neg = [], [], []

    for idx, layer_name in enumerate(layer_names):
        W = weights_dict[layer_name]
        W_pos = np.maximum(W, 0.0)
        W_neg = np.maximum(-W, 0.0)

        if len(layer_names) == 1:
            src_name = 'input'
            dst_name = 'output'
        elif idx == 0:
            src_name = 'input'
            dst_name = 'layer1'
        elif idx == len(layer_names) - 1:
            src_name = f'layer{idx}'
            dst_name = 'output'
        else:
            src_name = f'layer{idx}'
            dst_name = f'layer{idx+1}'

        i, j = np.nonzero(W_pos)
        if len(i) > 0:
            rows_pos.extend(offsets[src_name] + i)
            cols_pos.extend(offsets[dst_name] + j)
            data_pos.extend(W_pos[i, j].astype(np.float32))

        i, j = np.nonzero(W_neg)
        if len(i) > 0:
            rows_neg.extend(offsets[src_name] + i)
            cols_neg.extend(offsets[dst_name] + j)
            data_neg.extend(W_neg[i, j].astype(np.float32))

    if data_pos:
        adj_pos = csr_matrix(
            (data_pos, (rows_pos, cols_pos)),
            shape=(total_nodes, total_nodes),
            dtype=np.float32,
        )
    else:
        adj_pos = csr_matrix((total_nodes, total_nodes), dtype=np.float32)

    if data_neg:
        adj_neg = csr_matrix(
            (data_neg, (rows_neg, cols_neg)),
            shape=(total_nodes, total_nodes),
            dtype=np.float32,
        )
    else:
        adj_neg = csr_matrix((total_nodes, total_nodes), dtype=np.float32)

    return adj_pos, adj_neg


def convert_epoch_stage(epoch_dir, stage, output_base_dir, binary=False):
    stage_dir = os.path.join(epoch_dir, stage)

    weights_dict, available_layers = load_weights_from_directory(stage_dir)

    if not available_layers:
        print(f"  Skipping {stage_dir}: no weight layers found")
        return False

    adj_matrix = build_adjacency_matrix(weights_dict, available_layers, binary=binary)

    epoch_name = os.path.basename(epoch_dir)
    output_dir = os.path.join(output_base_dir, epoch_name)
    os.makedirs(output_dir, exist_ok=True)

    mode = 'binary' if binary else 'weighted'
    output_file = os.path.join(output_dir, f"{stage}_{mode}.npz")
    save_npz(output_file, adj_matrix)

    return True


def convert_epoch_stage_dual(epoch_dir, stage, output_base_dir):
    stage_dir = os.path.join(epoch_dir, stage)
    weights_dict, available_layers = load_weights_from_directory(stage_dir)
    if not available_layers:
        return False
    adj_pos, adj_neg = build_adjacency_matrix_dual(weights_dict, available_layers)
    epoch_name = os.path.basename(epoch_dir)
    output_dir = os.path.join(output_base_dir, epoch_name)
    os.makedirs(output_dir, exist_ok=True)
    save_npz(os.path.join(output_dir, f"{stage}_positive.npz"), adj_pos)
    save_npz(os.path.join(output_dir, f"{stage}_negative.npz"), adj_neg)
    return True


def process_snapshots_dir(input_dir, output_dir, verbose=True):
    epoch_dirs = sorted([d for d in os.listdir(input_dir)
                        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('epoch_')])
    for epoch_dir_name in epoch_dirs:
        epoch_dir = os.path.join(input_dir, epoch_dir_name)
        epoch_num = epoch_dir_name.split('_')[1]
        if verbose:
            print(f"  Epoch {epoch_num}...")
        for binary in [False, True]:
            convert_epoch_stage(epoch_dir, 'after_training', output_dir, binary=binary)
            convert_epoch_stage(epoch_dir, 'after_pruning', output_dir, binary=binary)
        convert_epoch_stage_dual(epoch_dir, 'after_training', output_dir)
        convert_epoch_stage_dual(epoch_dir, 'after_pruning', output_dir)
    return len(epoch_dirs)

RESULTS_DIR = "SET-MLP-Keras-Weights-Mask/results"

SOURCES = {
    "set_mlp": (
        os.path.join(RESULTS_DIR, "graph_snapshots"),
        os.path.join(RESULTS_DIR, "adjacency_matrices"),
    ),
    "rbm": (
        os.path.join(RESULTS_DIR, "graph_snapshots_rbm"),
        os.path.join(RESULTS_DIR, "adjacency_matrices_rbm"),
    ),
    "static": (
        os.path.join(RESULTS_DIR, "graph_snapshots_static"),
        os.path.join(RESULTS_DIR, "adjacency_matrices_static"),
    ),
}



def process_source(name, input_base, output_base):
    if not os.path.isdir(input_base):
        print(f"Skipping {name}: {input_base} not found")
        return

    print(f"\nProcessing [{name}]: {input_base} -> {output_base}")
    os.makedirs(output_base, exist_ok=True)

    run_dirs = sorted([d for d in os.listdir(input_base)
                      if os.path.isdir(os.path.join(input_base, d)) and d.startswith('run_')])

    if run_dirs:
        for run_name in run_dirs:
            run_input = os.path.join(input_base, run_name)
            run_output = os.path.join(output_base, run_name)
            print(f"  Run {run_name}...")
            n_epochs = process_snapshots_dir(run_input, run_output)
            print(f"    -> {n_epochs} epochs in {run_output}")
    else:
        print("  Single model (no run_* dirs)...")
        n_epochs = process_snapshots_dir(input_base, output_base)
        print(f"    -> {n_epochs} epochs in {output_base}")


def main():
    global SOURCE
    parser = argparse.ArgumentParser(
        description="Convert graph snapshots to adjacency matrices."
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Snapshot source to convert: set_mlp, rbm, static, or all.",
    )
    parser.add_argument(
        "legacy_source",
        nargs="?",
        default=None,
        help="Legacy positional source argument (set_mlp|rbm|static|all).",
    )
    args = parser.parse_args()

    if args.snapshot:
        SOURCE = args.snapshot.strip().lower()
    elif args.legacy_source:
        SOURCE = args.legacy_source.strip().lower()

    if SOURCE == "all":
        for name, (input_base, output_base) in SOURCES.items():
            process_source(name, input_base, output_base)
    elif SOURCE in SOURCES:
        input_base, output_base = SOURCES[SOURCE]
        process_source(SOURCE, input_base, output_base)
    else:
        print(f"Unknown source: '{SOURCE}'. Available: {list(SOURCES.keys())} or 'all'")
        print("Usage: python convert_to_adjacency.py --snapshot [set_mlp|rbm|static|all]")
        print("Legacy usage: python convert_to_adjacency.py [set_mlp|rbm|static|all]")


if __name__ == '__main__':
    main()
