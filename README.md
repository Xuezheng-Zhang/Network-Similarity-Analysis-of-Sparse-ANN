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

- `experiment/` – training scripts for running experiments on servers.
- `script/` – training scripts for running experiments on servers.
- `similarity_metrics/` – implementation of graph similarity metrics
- `SET-MLP-Keras-Weights-Mask/` – main code for sparse ANN training, conversion, and analysis.

## Quick Start

1. Run training for your custom sparsity setting:
```bash
python3 SET-MLP-Keras-Weights-Mask/set_mlp_keras_cifar10.py
```
2. Convert training outputs to adjacency matrices:
```bash
python3 SET-MLP-Keras-Weights-Mask/convert_to_adjacency.py
```
3. Run similarity analysis:

```bash
python3 SET-MLP-Keras-Weights-Mask/analyze_similarity.py \
  --experiment-root SET-MLP-Keras-Weights-Mask/results \
  --n 10 --matrix-type dual
```

Parameter notes:
- `--experiment-root`: root directory 
- `--n`: epoch step size 
- `--matrix-type`: modes used for similarity (`binary`, `weighted`, or `dual`).

4. Generate similarity evolution plots:

```bash
python3 SET-MLP-Keras-Weights-Mask/plot_similarity.py set_mlp dual
```

Parameter notes:
- `set_mlp`: data source of different models to plot (`set_mlp`, `rbm`, or `static`).
- `dual`: matrix type to plot (`binary`, `weighted`, `dual`, or `all`).

