import numpy as np
import os
import sys
from scipy.sparse import csr_matrix, save_npz

def load_weights_from_directory(stage_dir):
    """
    Load weight matrices from directory
    """
    weights = {}
    for layer in [1, 2, 3, 4]:
        weight_file = os.path.join(stage_dir, f"weight_layer_{layer}.npy")
        if os.path.exists(weight_file):
            weights[f'layer_{layer}'] = np.load(weight_file)
        else:
            print(f"Weight file not found: {weight_file}")
    return weights

def calculate_layer_sizes(weights_dict):
    """
    Calculate number of nodes per layer from weight matrix shapes
    """
    W1 = weights_dict['layer_1']  
    W2 = weights_dict['layer_2']  
    W3 = weights_dict['layer_3']  
    W4 = weights_dict['layer_4']
    
    # Extract layer sizes from weight matrix shapes
    input_size = W1.shape[0]
    layer1_size = W1.shape[1]  
    layer2_size = W2.shape[1]  
    layer3_size = W3.shape[1] 
    output_size = W4.shape[1]
    
    nodes_per_layer = {
        'input': input_size,
        'layer1': layer1_size,
        'layer2': layer2_size,
        'layer3': layer3_size,
        'output': output_size
    }
    
    # Calculate offsets
    offsets = {
        'input': 0,
        'layer1': input_size,
        'layer2': input_size + layer1_size,
        'layer3': input_size + layer1_size + layer2_size,
        'output': input_size + layer1_size + layer2_size + layer3_size
    }
    
    total_nodes = sum(nodes_per_layer.values())
    
    return nodes_per_layer, offsets, total_nodes

def build_adjacency_matrix(weights_dict, binary=False):
    """
    Build adjacency matrix from weight matrices
    
    Parameters:
        weights_dict: Dictionary of weight matrices
        binary: If True, create binary adjacency matrix; If False, use weight values

    Returns:
        Adjacency matrix
    """
    # Calculate layer sizes and offsets dynamically
    _, offsets, total_nodes = calculate_layer_sizes(weights_dict)
    
    rows = []
    cols = []
    data = []
    
    # Layer 1: input -> layer1
    W1 = weights_dict['layer_1']
    if binary:
        i, j = np.nonzero(W1)
        rows.extend(offsets['input'] + i)
        cols.extend(offsets['layer1'] + j)
        data.extend(np.ones(len(i), dtype=np.float32))
    else:
        i, j = np.nonzero(W1)
        rows.extend(offsets['input'] + i)
        cols.extend(offsets['layer1'] + j)
        data.extend(W1[i, j].astype(np.float32))
    
    # Layer 2: layer1 -> layer2
    W2 = weights_dict['layer_2']
    if binary:
        i, j = np.nonzero(W2)
        rows.extend(offsets['layer1'] + i)
        cols.extend(offsets['layer2'] + j)
        data.extend(np.ones(len(i), dtype=np.float32))
    else:
        i, j = np.nonzero(W2)
        rows.extend(offsets['layer1'] + i)
        cols.extend(offsets['layer2'] + j)
        data.extend(W2[i, j].astype(np.float32))
    
    # Layer 3: layer2 -> layer3
    W3 = weights_dict['layer_3']
    if binary:
        i, j = np.nonzero(W3)
        rows.extend(offsets['layer2'] + i)
        cols.extend(offsets['layer3'] + j)
        data.extend(np.ones(len(i), dtype=np.float32))
    else:
        i, j = np.nonzero(W3)
        rows.extend(offsets['layer2'] + i)
        cols.extend(offsets['layer3'] + j)
        data.extend(W3[i, j].astype(np.float32))
    
    # Layer 4: layer3 -> output
    W4 = weights_dict['layer_4']
    if binary:
        i, j = np.nonzero(W4)
        rows.extend(offsets['layer3'] + i)
        cols.extend(offsets['output'] + j)
        data.extend(np.ones(len(i), dtype=np.float32))
    else:
        i, j = np.nonzero(W4)
        rows.extend(offsets['layer3'] + i)
        cols.extend(offsets['output'] + j)
        data.extend(W4[i, j].astype(np.float32))
    
    # Build sparse matrix
    adj_matrix = csr_matrix((data, (rows, cols)), 
                            shape=(total_nodes, total_nodes), 
                            dtype=np.float32)
    return adj_matrix

def convert_epoch_stage(epoch_dir, stage, output_base_dir, binary=False):
    """
    Convert weights to adjacency matrix for a specific epoch and stage
    """
    stage_dir = os.path.join(epoch_dir, stage)
    
    # Load weights
    weights_dict = load_weights_from_directory(stage_dir)
    
    # Build adjacency matrix
    adj_matrix = build_adjacency_matrix(weights_dict, binary=binary)
    
    # Create output directory
    epoch_name = os.path.basename(epoch_dir)
    output_dir = os.path.join(output_base_dir, epoch_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as npz
    mode = 'binary' if binary else 'weighted'
    output_file = os.path.join(output_dir, f"{stage}_{mode}.npz")
    save_npz(output_file, adj_matrix)
    
    return True

def main():
    # Input and output directories
    input_base_dir = "SET-MLP-Keras-Weights-Mask/results/graph_snapshots"
    output_base_dir = "SET-MLP-Keras-Weights-Mask/results/adjacency_matrices"
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all epochs
    epoch_dirs = sorted([d for d in os.listdir(input_base_dir) 
                        if os.path.isdir(os.path.join(input_base_dir, d)) and d.startswith('epoch_')])
    
    # Process each epoch
    for epoch_dir_name in epoch_dirs:
        epoch_dir = os.path.join(input_base_dir, epoch_dir_name)
        epoch_num = epoch_dir_name.split('_')[1]
        print(f"Processing Epoch {epoch_num}...")
        
        # Process after_training
        for binary in [False, True]:
            convert_epoch_stage(epoch_dir, 'after_training', output_base_dir, binary=binary)
        
        # Process after_pruning
        for binary in [False, True]:
            convert_epoch_stage(epoch_dir, 'after_pruning', output_base_dir, binary=binary)

if __name__ == '__main__':
    main()
