import json
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "SET-MLP-Keras-Weights-Mask/results"
RUN_ID = 0

if RUN_ID is not None:
    metadata_path = os.path.join(RESULTS_DIR, f"training_metadata_static_run_{RUN_ID}.json")
else:
    metadata_path = os.path.join(RESULTS_DIR, "training_metadata.json")

with open(metadata_path) as f:
    records = json.load(f)
epochs = np.array([r["epoch"] for r in records])
val_accuracy = np.array([r["val_accuracy"] for r in records])
val_loss = np.array([r["val_loss"] for r in records])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(epochs, val_accuracy * 100, 'r-o', label='SET-MLP', linewidth=2, markersize=1)
ax1.set_xlabel('Epochs[#]', fontsize=12)
ax1.set_ylabel('Accuracy [%]', fontsize=12)
ax1.set_title(f'SET-MLP Training: Validation Accuracy', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, val_loss, 'b-o', label='SET-MLP', linewidth=2, markersize=1)
ax2.set_xlabel('Epochs[#]', fontsize=12)
ax2.set_ylabel('Validation Loss', fontsize=12)
ax2.set_title(f'SET-MLP Training: Validation Loss', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
if RUN_ID is not None:
    out_name = f"set_mlp_training_performance_run_{RUN_ID}.png"
else:
    out_name = "set_mlp_training_performance.png"
out_path = os.path.join(RESULTS_DIR, out_name)
plt.savefig(out_path, dpi=300)
print(f"Performance plot saved to {out_path}")
plt.close()