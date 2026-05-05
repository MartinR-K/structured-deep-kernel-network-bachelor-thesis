# Script zum plotten der ROC Kurven
# Dateien vom Format .pt im ordner "ROCs" werden gelesen und geplottet
# Der Plot wird in "ROCs" gespeichert
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

run_dir = "ROCs"

# Titel der figure, Namen der Dateien, Namen der plots
title = "Test ROC curves of NNs on Higgs dataset"
files = ["[2000]","[1600, 1600]","[1200, 1200, 1200, 1200]","[1200, 1200, 1200, 1200, 1200, 1200]","[1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600]"]
names = ["1x2000", "2x1600","4x1200","6x1200","8x1600"]

plt.figure()

for i, file in enumerate(files):
    data = torch.load(os.path.join(run_dir, f"test_roc_{file}.pt"))
    targets = data["targets"].numpy().ravel()
    probs = data["probs"].numpy().ravel()

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{names[i]} (AUC = {roc_auc:.3f})")

# Diagonale
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.title(title)
plt.tight_layout()

# Plot-Name aus allen Namen bauen
plot_name = "ROCplot_" + "_".join(files) + ".pdf"
plot_path = os.path.join(run_dir, plot_name)

plt.savefig(plot_path)
plt.show()

print(f"[INFO] ROC plot saved to: {plot_path}")