import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

metadata = pd.read_csv("SET-MLP-Keras-Weights-Mask/results/training_metadata.txt", delimiter='\t')

epochs = metadata['Epoch'].values
val_accuracy = metadata['Val_Accuracy'].values
val_loss = metadata['Val_Loss'].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(epochs, val_accuracy * 100, 'r-o', label='SET-MLP', linewidth=2, markersize=5)
ax1.set_xlabel('Epochs[#]', fontsize=12)
ax1.set_ylabel('CIFAR10\nAccuracy [%]', fontsize=12)
ax1.set_title('SET-MLP Training: Validation Accuracy', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, val_loss, 'b-o', label='SET-MLP', linewidth=2, markersize=5)
ax2.set_xlabel('Epochs[#]', fontsize=12)
ax2.set_ylabel('Validation Loss', fontsize=12)
ax2.set_title('SET-MLP Training: Validation Loss', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("SET-MLP-Keras-Weights-Mask/results/set_mlp_training_performance.pdf", dpi=300)
plt.savefig("SET-MLP-Keras-Weights-Mask/results/set_mlp_training_performance.png", dpi=300)
print("Performance plot saved to SET-MLP-Keras-Weights-Mask/results/set_mlp_training_performance.pdf and .png")
plt.close()
    