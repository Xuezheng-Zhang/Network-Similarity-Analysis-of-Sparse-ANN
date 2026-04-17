# Network Similarity Analysis of Sparse ANN

The project aims to study and analyze the structural evolution trends of sparse ANN during the training process.

## Project Objectives
- Implement the calculation of various graph similarity metrics:
  - DeltaCon
  - Jaccard
- Train sparse neural networks and save its graph snapshots.
- Convert networks to adjacency matrices, we have three mode:
  - binary
  - weighted
  - dual
- Compute graph similarity metrics between different epochs.
- Plot various similarity evolution diagrams for analyzing.

## Project structure

```text
Network-Similarity-Analysis-of-Sparse-ANN/
├── experiment/                              # Server-side experiment scripts
├── script/                                  # Server-side config scripts
├── similarity_metrics/                      # Graph similarity implementations
└── SET-MLP-Keras-Weights-Mask/
    ├── set_mlp_keras_cifar10.py             # SET-MLP training
    ├── static_mlp_keras_cifar10.py          # Static Sparse MLP training 
    ├── set_rbm.py                           # SET-RBM training 
    ├── convert_to_adjacency.py              # Convert snapshots to adjacency matrices
    ├── analyze_similarity.py                # Compute Dual DeltaCon/Jaccard similarities
    ├── analyze_networks.py                  # Analyze network properties
    ├── weight_sign_stats.py                 # Compute positive/negative weight
    ├── analyze_random_similarity.py         # Compar Random-graph similarity
    ├── plot_similarity.py                   # Plot similarity evolution
    ├── plot_performance.py                  # Plot training performance
    └── results/                             # Metadata, CSV files, and diagrams
```

## Quick Start
Take the analysis of the similarity of set-mlp model as an example:

### 1. Run training for your custom sparsity setting:
```bash
python3 SET-MLP-Keras-Weights-Mask/set_mlp_keras_cifar10.py
```
After this step, outputs are saved in `SET-MLP-Keras-Weights-Mask/results/` include:
- `graph_snapshots/` (or other similar snapshot folders): model weights for each epoch.
- `training_metadata_run_*.json`: training data for each epoch,such as accuracy and loss.

### 2. Convert training snapshots to adjacency matrices:
```bash
python3 SET-MLP-Keras-Weights-Mask/convert_to_adjacency.py --snapshot set_mlp
```
After this step, outputs are saved in `SET-MLP-Keras-Weights-Mask/results/adjacency_matrices/`:
- `run_*/epoch_xxxx/after_training_*.npz`
- `run_*/epoch_xxxx/after_pruning_*.npz`

Parameter notes:
-  `--snapshot`: snapshot type (`set_mlp`, `rbm`, `static`, or `all`, default is `all`).

### 3. Run similarity analysis:

```bash
python3 SET-MLP-Keras-Weights-Mask/analyze_similarity.py \
   --experiment-root SET-MLP-Keras-Weights-Mask/results/adjacency_matrices \
  --n 10 \
  --matrix-type dual
```

After this step, outputs are saved in `SET-MLP-Keras-Weights-Mask/results/similarity/` include:
- `deltacon_similarity_*.csv`: Dual DeltaCon similarity results between epochs.
- `jaccard_similarity_*.csv`: Jaccard similarity results between epochs.

Parameter notes:
- `--experiment-root`: root directory of analysis data.
- `--n`: epoch step size (for example, `n=10` compare `t` vs `t+10`, default is `10`).
- `--g`: DeltaCon parameter `g` (default is `5`).
- `--matrix-type`: matrix type for similarity calculation (`binary`, `weighted`, or `dual`, default is `dual`).

### 4. Generate similarity analyze plots:

```bash
python3 SET-MLP-Keras-Weights-Mask/plot_similarity.py set_mlp dual
```

After this step, plots are saved in `SET-MLP-Keras-Weights-Mask/results/diagram/`.

Parameter notes:
- `set_mlp`: data source of different models to plot (`set_mlp`, `rbm`, or `static`).
- `dual`: matrix type to plot (`binary`, `weighted`, `dual`, or `all`).

