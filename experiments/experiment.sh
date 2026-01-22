#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --account=cseduproject
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=./logs/experiment1_%j.out
#SBATCH --error=./logs/experiment1_%j.err
#SBATCH --mail-user=xuezheng.zhang@ru.nl


VENV_PATH="/scratch/xzhang/virtual_environments/bin/activate"
PROJECT_DIR="/home/xzhang/intern/Network-Similarity-Analysis-of-Sparse-ANN"

source "$VENV_PATH"
python3 "$PROJECT_DIR/SET-MLP-Keras-Weights-Mask/set_mlp_keras_cifar10.py"