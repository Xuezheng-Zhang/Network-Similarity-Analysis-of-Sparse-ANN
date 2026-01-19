import numpy as np
import os
import sys
from datetime import datetime
from scipy.sparse import load_npz

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

if __name__ == '__main__':
    # Create output file
    output_filename = "SET-MLP-Keras-Weights-Mask/results/network_analysis.txt"
    os.makedirs("SET-MLP-Keras-Weights-Mask/results", exist_ok=True)
    
    with open(output_filename, 'w') as output_file:
        
        # Check if command line arguments exist
        if len(sys.argv) > 1:
            # Directly test specified file
            test_npy_file(sys.argv[1], output_file)
        else:
            base_dir = "SET-MLP-Keras-Weights-Mask/results/graph_snapshots"
            
            if not os.path.exists(base_dir):
                output_file.write(f"Directory does not exist: {base_dir}\n")
                sys.exit(1)
            
            # Find all epochs
            epoch_dirs = sorted([d for d in os.listdir(base_dir) 
                                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('epoch_')])
            
            if not epoch_dirs:
                output_file.write(f"No epoch directories found\n")
                sys.exit(1)
            
            # Calculate total elements for all epochs
            output_file.write(f"Total Elements Summary (All Layers Combined)\n")
            output_file.write(f"{'Epoch':<10} {'Stage':<20} {'Non-zero':<15} {'Zero':<15} {'Total':<15} {'Sparsity':<10}\n")
            output_file.write(f"{'-'*70}\n")
            
            for epoch_dir_name in epoch_dirs:
                epoch_dir = os.path.join(base_dir, epoch_dir_name)
                epoch_num = epoch_dir_name.split('_')[1]
                
                # Calculate for after_training
                non_zero_train, zero_train, total_train = calculate_total_elements_per_epoch(epoch_dir, 'after_training')
                if non_zero_train is not None:
                    sparsity_train = (zero_train / total_train * 100) if total_train > 0 else 0
                    output_file.write(f"{epoch_num:<10} {'after_training':<20} {non_zero_train:<15,} {zero_train:<15,} {total_train:<15,} {sparsity_train:<10.2f}%\n")
                
                # Calculate for after_pruning
                non_zero_prune, zero_prune, total_prune = calculate_total_elements_per_epoch(epoch_dir, 'after_pruning')
                if non_zero_prune is not None:
                    sparsity_prune = (zero_prune / total_prune * 100) if total_prune > 0 else 0
                    output_file.write(f"{epoch_num:<10} {'after_pruning':<20} {non_zero_prune:<15,} {zero_prune:<15,} {total_prune:<15,} {sparsity_prune:<10.2f}%\n")
            
            # test files from first epoch
            epoch_dir = os.path.join(base_dir, epoch_dirs[0])
            output_file.write(f"\n{'#'*70}\n")
            output_file.write(f"Detailed Analysis for Epoch: {epoch_dirs[0]}\n")
            
            # Test after_training files
            after_training = os.path.join(epoch_dir, 'after_training')
            if os.path.exists(after_training):
                output_file.write(f"{'-'*70}\n")
                output_file.write(f"# After Training Stage\n")
                
                # Test weight files
                for layer in [1, 2, 3, 4]:
                    weight_file = os.path.join(after_training, f"weight_layer_{layer}.npz")
                    if os.path.exists(weight_file):
                        test_npy_file(weight_file, output_file)
                
                # Test mask files
                for layer in [1, 2, 3]:
                    mask_file = os.path.join(after_training, f"mask_layer_{layer}.npz")
                    if os.path.exists(mask_file):
                        test_npy_file(mask_file, output_file)
            
            # Test after_pruning files
            after_pruning = os.path.join(epoch_dir, 'after_pruning')
            if os.path.exists(after_pruning):
                output_file.write(f"{'-'*70}\n")
                output_file.write(f"# After Pruning Stage\n")
                
                # Test weight files
                for layer in [1, 2, 3, 4]:
                    weight_file = os.path.join(after_pruning, f"weight_layer_{layer}.npz")
                    if os.path.exists(weight_file):
                        test_npy_file(weight_file, output_file)
                
                # Test mask files
                for layer in [1, 2, 3]:
                    mask_file = os.path.join(after_pruning, f"mask_layer_{layer}.npz")
                    if os.path.exists(mask_file):
                        test_npy_file(mask_file, output_file)
    
    print(f"Analysis complete! Results saved to: {output_filename}")
