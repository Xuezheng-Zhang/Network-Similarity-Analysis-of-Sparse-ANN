import numpy as np
import os
import sys
from scipy.sparse import load_npz

GRAPH_SNAPSHOTS_BASE = "SET-MLP-Keras-Weights-Mask/results/graph_snapshots"
RESULTS_DIR = "SET-MLP-Keras-Weights-Mask/results"

RUN_ID = 0              
OUTPUT_FILE = os.path.join(RESULTS_DIR, "network_analysis.txt")

def load_matrix_file(filepath):
    """
    Load matrix from .npz file
    """
    sparse_matrix = load_npz(filepath)
    return sparse_matrix.toarray()

def test_npy_file(filepath, output_file):
    """Test a single npz file"""
    output_file.write(f"File: {filepath}\n")
    
    if not os.path.exists(filepath):
        output_file.write(f"File does not exist!\n")
        return
    
    data = load_matrix_file(filepath)
    
    output_file.write(f"Shape: {data.shape}\n")
    output_file.write(f"Data type: {data.dtype}\n")
    output_file.write(f"Total elements: {data.size:,}\n")
    
    if data.dtype in [np.float32, np.float64]:
        output_file.write(f"\nStatistics:\n")
        output_file.write(f"  Min: {np.min(data):.6f}\n")
        output_file.write(f"  Max: {np.max(data):.6f}\n")
        output_file.write(f"  Mean: {np.mean(data):.6f}\n")
        output_file.write(f"  Std: {np.std(data):.6f}\n")
    
    non_zero = np.count_nonzero(data)
    sparsity = (data.size - non_zero) / data.size
    output_file.write(f"\nSparsity:\n")
    output_file.write(f"  Non-zero elements: {non_zero:,} ({non_zero/data.size*100:.2f}%)\n")
    output_file.write(f"  Zero elements: {data.size - non_zero:,} ({sparsity*100:.2f}%)\n")

def calculate_total_elements_per_epoch(epoch_dir, stage):
    """
    Calculate total non-zero and zero elements across all layers for a given epoch and stage
    
    Parameters:
        epoch_dir: Path to epoch directory
        stage: 'after_training' or 'after_pruning'
    
    Returns:
        tuple: (total_non_zero, total_zero, total_elements)
    """
    stage_dir = os.path.join(epoch_dir, stage)
    if not os.path.exists(stage_dir):
        return None, None, None
    
    total_non_zero = 0
    total_zero = 0
    total_elements = 0
    
    # Calculate for all weight layers (1-4)
    for layer in [1, 2, 3, 4]:
        weight_file = os.path.join(stage_dir, f"weight_layer_{layer}.npz")
        
        if os.path.exists(weight_file):
            data = load_matrix_file(weight_file)
            non_zero = np.count_nonzero(data)
            total_non_zero += non_zero
            total_zero += (data.size - non_zero)
            total_elements += data.size
    
    return total_non_zero, total_zero, total_elements

def analyze_run(base_dir, output_file, detailed_epoch=None):
    """
    Analyze network data from a given base directory (e.g. graph_snapshots/run_0).
    
    Parameters:
        base_dir: Path to directory containing epoch_* subdirs
        output_file: File handle to write output
        detailed_epoch: If set, add detailed analysis for this epoch (e.g. 'epoch_0').
                        If None, use first epoch.
    """
    if not os.path.exists(base_dir):
        output_file.write(f"Directory does not exist: {base_dir}\n")
        return
    
    epoch_dirs = sorted([d for d in os.listdir(base_dir) 
                        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('epoch_')])
    
    if not epoch_dirs:
        output_file.write(f"No epoch directories found in {base_dir}\n")
        return
    
    output_file.write(f"Total Elements Summary (All Layers Combined)\n")
    output_file.write(f"Source: {base_dir}\n")
    output_file.write(f"{'Epoch':<10} {'Stage':<20} {'Non-zero':<15} {'Zero':<15} {'Total':<15} {'Sparsity':<10}\n")
    output_file.write(f"{'-'*70}\n")
    
    for epoch_dir_name in epoch_dirs:
        epoch_dir = os.path.join(base_dir, epoch_dir_name)
        epoch_num = epoch_dir_name.split('_')[1]
        
        non_zero_train, zero_train, total_train = calculate_total_elements_per_epoch(epoch_dir, 'after_training')
        if non_zero_train is not None:
            sparsity_train = (zero_train / total_train * 100) if total_train > 0 else 0
            output_file.write(f"{epoch_num:<10} {'after_training':<20} {non_zero_train:<15,} {zero_train:<15,} {total_train:<15,} {sparsity_train:<10.2f}%\n")
        
        non_zero_prune, zero_prune, total_prune = calculate_total_elements_per_epoch(epoch_dir, 'after_pruning')
        if non_zero_prune is not None:
            sparsity_prune = (zero_prune / total_prune * 100) if total_prune > 0 else 0
            output_file.write(f"{epoch_num:<10} {'after_pruning':<20} {non_zero_prune:<15,} {zero_prune:<15,} {total_prune:<15,} {sparsity_prune:<10.2f}%\n")
    
    epoch_for_detail = detailed_epoch or epoch_dirs[0]
    epoch_dir = os.path.join(base_dir, epoch_for_detail)
    output_file.write(f"\n{'#'*70}\n")
    output_file.write(f"Detailed Analysis for Epoch: {epoch_for_detail}\n")
    
    for stage in ['after_training', 'after_pruning']:
        stage_dir = os.path.join(epoch_dir, stage)
        if not os.path.exists(stage_dir):
            continue
        output_file.write(f"{'-'*70}\n")
        output_file.write(f"# {stage.replace('_', ' ').title()}\n")
        
        for layer in [1, 2, 3, 4]:
            weight_file = os.path.join(stage_dir, f"weight_layer_{layer}.npz")
            if os.path.exists(weight_file):
                test_npy_file(weight_file, output_file)
        
        for layer in [1, 2, 3]:
            mask_file = os.path.join(stage_dir, f"mask_layer_{layer}.npz")
            if os.path.exists(mask_file):
                test_npy_file(mask_file, output_file)

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(OUTPUT_FILE, 'w') as output_file:
        base = GRAPH_SNAPSHOTS_BASE
        if not os.path.exists(base):
            output_file.write(f"Directory does not exist: {base}\n")
            sys.exit(1)

        run_dirs = sorted([d for d in os.listdir(base)
                          if os.path.isdir(os.path.join(base, d)) and d.startswith('run_')])

        if RUN_ID is not None:
            run_name = f"run_{RUN_ID}"
            run_path = os.path.join(base, run_name)
            if not os.path.exists(run_path):
                output_file.write(f"Run {RUN_ID} not found: {run_path}\n")
                sys.exit(1)
            output_file.write(f"Analyzing run {RUN_ID}\n\n")
            analyze_run(run_path, output_file)
        elif run_dirs:
            for run_name in run_dirs:
                run_path = os.path.join(base, run_name)
                rid = run_name.replace('run_', '')
                output_file.write(f"\n{'='*70}\n")
                output_file.write(f"Run {rid}\n")
                output_file.write(f"{'='*70}\n\n")
                analyze_run(run_path, output_file)
        else:
            output_file.write("Single run (legacy format)\n\n")
            analyze_run(base, output_file)

    print(f"Analysis complete! Results saved to: {OUTPUT_FILE}")
