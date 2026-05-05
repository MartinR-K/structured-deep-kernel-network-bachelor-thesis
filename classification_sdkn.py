# classification_sdkn.py 
# Diese script trainiert ein SDKN
# auf den Higgs Datensatz.
# Vor Ausführung bitte sicher stellen:
# Die Klasse SDKN in utils/lightning_modules.py muss von "BinaryClassificationNetwork" erben.
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch
torch.set_float32_matmul_precision("medium")  # Tensor Cores

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, auc

from utils import lightning_models
from utils.utilities import (
    load_full_dataset,
    train_val_test_split_dataset,
    normalize_train_only,
    make_loader,
    LossHistory,
    compute_centers
)



# Settings
DATA_FILE = "HIGGS.csv"

M = 5
hidden_dims = [1600,1600,1600,1600,1600,1600]
L = len(hidden_dims)

d0 = 28 # input dimension
d = 1 # output dimension

max_epochs = 50
batch_size = 4096
num_workers = 8
GRID_ROWS = "all"  # "all" für gesamten trainingsdatensatz, 1_000_000 für random subset der Größe 1 Mio

name = f"M={M}_{hidden_dims}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[INFO] hyperparameters: L={L}, M={M}, hidden dimensions={hidden_dims}\n")


# Create unique run directory
timestamp = datetime.now().strftime("%d-%m_%H-%M")
run_dir = os.path.join("results", "Higgs", "single_model", f"higgs_sdkn_{name}_{timestamp}")
os.makedirs(run_dir, exist_ok=False)
print(f"[INFO] Run directory: {run_dir}")



# Load full dataset
X_full, y_full = load_full_dataset(DATA_FILE)

(
    X_train_full, y_train_full, 
    X_val_full, y_val_full, 
    X_test_full, y_test_full
) = train_val_test_split_dataset(
    X_full, y_full,
    test_size=0.1,
    val_size=0.1,
    stratify=True
)

print("[INFO] Dataset sizes:")
print(f"  Train: {len(X_train_full)}")
print(f"  Val:   {len(X_val_full)}")
print(f"  Test:  {len(X_test_full)}\n")



# Subsample
if isinstance(GRID_ROWS,int):
    print(f"[INFO] Using subset of {GRID_ROWS} samples")
    indices = torch.randperm(len(X_train_full))[:GRID_ROWS]
    X_train_full = X_train_full[indices]
    y_train_full = y_train_full[indices]



# Training

(
    X_train_n,
    y_train_n,
    X_val_n,
    y_val_n,
    X_test_n,
    y_test_n,
    mean_x,
    std_x,
    _,_,
) = normalize_train_only(X_train_full, y_train_full, (X_val_full, y_val_full), (X_test_full, y_test_full), only_features=True)

train_loader_full = make_loader(X_train_n, y_train_n, batch_size, shuffle=True)
val_loader = make_loader(X_val_n, y_val_n, batch_size, shuffle=False)
test_loader = make_loader(X_test_n, y_test_n, batch_size, shuffle=False)

# Random centers
centers = compute_centers(train_loader_full, method="random", num_centers=M)

model = lightning_models.SDKN(
    centers=centers, 
    L=L, 
    hidden_dims=hidden_dims,
    d0=d0, 
    d=d, 
    mean_x=mean_x, 
    std_x=std_x,
    roc_save_path=run_dir,
).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

loss_history = LossHistory(roc_save_path=run_dir)

# Callbacks: EarlyStopping + Best Checkpoint
final_ckpt = os.path.join(run_dir, "final_best.ckpt")
checkpoint_cb = ModelCheckpoint(
    monitor="val/loss", mode="min", save_top_k=1,
    dirpath=run_dir, filename="final_best"
)

early_stop_final = EarlyStopping(monitor="val/loss", patience=10, mode="min")

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu" if DEVICE == "cuda" else "cpu",
    devices=1,
    precision=16,
    callbacks=[early_stop_final, checkpoint_cb, loss_history],
    enable_progress_bar=True
)

trainer.fit(model, train_loader_full, val_loader)

# Load best checkpoint
best_final_model = lightning_models.SDKN.load_from_checkpoint(
    final_ckpt,
    centers=centers,
    L=L,
    hidden_dims=hidden_dims,
    d0=d0,
    d=d,
    mean_x=mean_x,
    std_x=std_x,
    roc_save_path=run_dir,
).to(DEVICE)

# Test
test_metrics = trainer.test(best_final_model, test_loader)
print(f"[INFO] Test loss: {test_metrics[0]['test/loss']}")



# Save final model
torch.save(best_final_model.state_dict(), os.path.join(run_dir, "final_model.pt"))

# --- Hyperparameter speichern ---
hparams_path = os.path.join(run_dir, f"hparams_{name}.json")
hparams = {
    "Datensatz": DATA_FILE,
    "P": num_params,
    "L": L,
    "M": M,
    "hidden_dims": hidden_dims,
    "d0": d0,
    "d": d,
    "mean_x": mean_x.tolist(),  # Tensor -> list
    "std_x": std_x.tolist(),
}
with open(hparams_path, "w") as f:
    json.dump(hparams, f)



# Save test_loss
test_loss = {
            "Datensatz": DATA_FILE,
            "P": num_params,
            "L": L,
            "M": M,
            "hidden_dims" :hidden_dims,
            "test_loss": test_metrics[0]["test/loss"],
            "test_auc": test_metrics[0]["test/auc"],
            "test_acc": test_metrics[0]["test/acc"],
            "test_best_threshold": test_metrics[0]["test/best_threshold"],
            "max_epochs": max_epochs,
            "batch_size": batch_size,
        }
df = pd.DataFrame([test_loss])
grid_path = os.path.join(run_dir, f"test_loss_higgs_{name}.csv")
df.to_csv(grid_path, index=False)
print(f"[INFO] Final test/loss saved: {grid_path}")



# Save loss history
# loss_history.train_losses und val_losses in dict
train_dict = {
    "train_losses": loss_history.train_losses,
    "val_losses": loss_history.val_losses,
    "val_aucs": loss_history.val_auc,
    "val_best_thresholds": loss_history.val_best_threshold,
    "val_accs": loss_history.val_acc
}

df = pd.DataFrame(train_dict)
grid_path = os.path.join(run_dir, f"history_higgs_{name}.csv")
df.to_csv(grid_path, index=False)
print(f"[INFO] Final train history saved: {grid_path}")



# Plot ROC
# passende Dateien finden
dataset_name = DATA_FILE
roc_file = next(f for f in os.listdir(run_dir) if f.startswith("test_roc_"))
roc_path = os.path.join(run_dir, roc_file)
data = torch.load(roc_path)

targets = data["targets"].numpy().ravel()
probs = data["probs"].numpy().ravel()

fpr, tpr, thresholds = roc_curve(targets, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.plot([],[], label=f"M = {M}", linestyle='None')
plt.plot([],[], label=f"hidden dimensions = {hidden_dims}", linestyle='None')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

# Titel nur mit Dataset und Hyperparameter
plt.title(
    "ROC on test data\n"
    f"SDKN with Dataset: {dataset_name}\n"
)
plt.grid()
plot_path = os.path.join(run_dir, "ROC.pdf")

plt.savefig(plot_path)
plt.show(block=False)


# Plot training curves
# Werte
history_path = os.path.join(run_dir, f"history_higgs_{name}.csv")
history = pd.read_csv(history_path)
grid_path = os.path.join(run_dir, f"test_loss_higgs_{name}.csv")
test_loss = pd.read_csv(grid_path)
train_losses = history.train_losses
val_losses = history.val_losses
val_aucs = history.val_aucs
val_best_thresholds = history.val_best_thresholds
val_accs = history.val_accs

best_val_loss_idx = val_losses.idxmin()
best_val_loss = val_losses.iloc[best_val_loss_idx]

best_val_auc_idx = val_aucs.idxmax()
best_val_auc = val_aucs.iloc[best_val_auc_idx]

best_val_acc_idx = val_accs.idxmax()
best_val_acc = val_accs.iloc[best_val_acc_idx]
best_val_best_threshold = val_best_thresholds[best_val_acc_idx]

best_test_loss = test_loss.test_loss[0]

epochs = np.arange(1, len(val_losses) + 1)
# plt.figure(figsize=(8,5))

# Plot der Loss-Kurven
# Grid
plt.figure(figsize=(8, 5))
plt.grid(True, linestyle='--', alpha=0.6)
# Beste Validation Loss/auc/acc markieren
plt.scatter(best_val_loss_idx + 1, best_val_loss, color='red', s=50, zorder=5)
plt.annotate(f"Min Val-Loss: {best_val_loss:.4f}",(best_val_loss_idx + 1, best_val_loss),textcoords="offset points", xytext=(0,10),ha='center', color='red')
plt.scatter(best_val_auc_idx + 1, best_val_auc, color='green', s=50, zorder=5, label=f"Max Val-AUC: {best_val_auc:.4f}")
plt.scatter(best_val_acc_idx + 1, best_val_acc, color='brown', s=50, zorder=5, label=f"Max Val-Acc: {best_val_acc:.4f}")
if best_val_loss_idx != best_val_auc_idx:
    plt.scatter(best_val_loss_idx + 1, val_aucs[best_val_loss_idx], color='purple', s=50, zorder=5, label=f"AUC = {val_aucs[best_val_loss_idx]:.4f}")
if best_val_loss_idx != best_val_acc_idx:
    plt.scatter(best_val_loss_idx + 1, val_accs[best_val_loss_idx], color='blue', s=50, zorder=5, label=f"Acc = {val_accs[best_val_loss_idx]:.4f}")

# Beste Test Loss in der Legende 
plt.plot([], [], linewidth=0, label=f"Test Loss: {best_test_loss:.4f}")

# 🔹 Kurven
plt.plot(epochs, train_losses, label="Train Loss", linestyle='-')
plt.plot(epochs, val_losses, label="Val Loss", linestyle='--')
plt.plot(epochs, val_aucs, label="Val AUROC", linestyle='-.')
plt.plot(epochs, val_accs, label="Val Accuracy", linestyle=':')


# Achsenbeschriftung
plt.xlabel("Epoch")
plt.ylabel("BCE-Loss, AUROC(AUC), Acc")

# Titel nur mit Dataset und Hyperparameter
plt.title(
    f"SDKN with Dataset: {dataset_name}\n"
    f"HPs: L={L}, M={M}, P={num_params}, batch size={batch_size}\n"
    f"hidden dimensions={hidden_dims}"
)

plt.legend(ncol=2)
plt.tight_layout()

# Speicherpfad
plot_path = os.path.join(run_dir, f"training_higgs_{name}.pdf")
plt.savefig(plot_path)
plt.show(block=False)

print(f"[INFO] Trainingsplot mit Markierungen gespeichert unter: {plot_path}")
