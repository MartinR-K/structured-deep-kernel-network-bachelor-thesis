# regression_nn.py 
# Diese script trainiert ein feedforward neural network mit ReLU activation 
# auf den Datensätzen "dataset_2175_kin8nm.arff" oder "airfoil_self_noise.dat".
# Vor Ausführung bitte sicher stellen:
# Die Klasse NN in utils/lightning_modules.py muss von "Network" erben.

import os
import json
import pandas as pd
import torch
torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import matplotlib.pyplot as plt

from utils import lightning_models
from utils.utilities import (
    load_full_dataset,
    train_val_test_split_dataset,
    normalize_train_only,
    make_loader,
    LossHistoryRegression
)


# Settings
target_params = 10000

n = 50
hidden_dims = [136,136]
L = len(hidden_dims)

d0 = 8 # input dimension
d = 1 # output dimension


num_epochs = 50
batch_size=12
name = f"{hidden_dims}"
DATA_FILE = "dataset_2175_kin8nm.arff" # "dataset_2175_kin8nm.arff", "airfoil_self_noise.dat"
if DATA_FILE.endswith(".dat"):
    DATA_NAME = "airfoil"
else:
    DATA_NAME = "kin8nm"

# Create unique run directory
timestamp = datetime.now().strftime("%d-%m_%H-%M")
run_dir = os.path.join("results", DATA_NAME, "single_model", f"{DATA_NAME}_nn_{name}_{timestamp}")
os.makedirs(run_dir, exist_ok=False)
print(f"[INFO] Run directory: {run_dir}")


# Load full dataset
X_full, y_full = load_full_dataset(DATA_FILE)

(
    X_train, y_train, 
    X_val, y_val, 
    X_test, y_test
) = train_val_test_split_dataset(
    X_full, y_full,
    test_size=0.1,
    val_size=0.1,
    stratify=False
)

print("[INFO] Dataset sizes:")
print(f"  Train: {len(X_train)}")
print(f"  Val:   {len(X_val)}")
print(f"  Test:  {len(X_test)}\n")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Training on full dataset
(
    X_train_n,
    y_train_n,
    X_val_n,
    y_val_n,
    X_test_n,
    y_test_n,
    mean_x,
    std_x,
    mean_y,
    std_y,
) = normalize_train_only(X_train, y_train, (X_val, y_val), (X_test, y_test))

train_loader_full = make_loader(X_train_n, y_train_n, batch_size, shuffle=True)
val_loader = make_loader(X_val_n, y_val_n, batch_size, shuffle=False)
test_loader = make_loader(X_test_n, y_test_n, batch_size, shuffle=False)

model = lightning_models.NN(
    L=L,
    hidden_dims=hidden_dims,
    d0=d0,
    d=d,
    mean_x=mean_x,
    std_x=std_x,
    mean_y=mean_y,
    std_y=std_y,
).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

loss_history = LossHistoryRegression()

ckpt = os.path.join(run_dir, "best.ckpt")
checkpoint_cb = ModelCheckpoint(
    monitor="val/loss", mode="min", save_top_k=1,
    dirpath=run_dir, filename="best"
)

early_stop_final = EarlyStopping(monitor="val/loss", patience=10, mode="min")

trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator="gpu" if DEVICE == "cuda" else "cpu",
    devices=1,
    callbacks=[early_stop_final, checkpoint_cb, loss_history],
)

trainer.fit(model, train_loader_full, val_loader)

# Load best model
best_final_model = lightning_models.NN.load_from_checkpoint(
    ckpt,
    L=L,
    hidden_dims=hidden_dims,
    d0=d0,
    d=d,
    mean_x=mean_x,
    std_x=std_x,
    mean_y=mean_y,
    std_y=std_y,
).to(DEVICE)

test_metrics = trainer.test(best_final_model, test_loader)
print("[INFO] Final Test Loss:", test_metrics[0]["test/loss"])


# Save final model
torch.save(best_final_model.state_dict(), os.path.join(run_dir, "final_model.pt"))


# --- Hyperparameter speichern ---
hparams_path = os.path.join(run_dir, "model_hparams.json")
hparams = {
    "Datensatz": DATA_FILE,
    "P": num_params,
    "L": L,
    "hidden_dims": hidden_dims,
    "d0": d0,
    "d": d,
    "mean_x": mean_x.tolist(),  # Tensor -> list
    "std_x": std_x.tolist(),
    "mean_y": mean_y.tolist(),
    "std_y": std_y.tolist(),
}
with open(hparams_path, "w") as f:
    json.dump(hparams, f)

print(f"[INFO] Modell + Hyperparameter gespeichert unter {run_dir}")

# Save performance of final model
test_loss = {
            "Datensatz": DATA_FILE,
            "P": num_params,
            "L": L,
            "hidden_dims": hidden_dims,
            "test_loss": test_metrics[0]["test/loss"],
            "max_epochs": num_epochs,
            "batch_size": batch_size,
        }
df = pd.DataFrame([test_loss])
grid_path = os.path.join(run_dir, f"test_loss_{DATA_NAME}_{name}.csv")
df.to_csv(grid_path, index=False)
print(f"[INFO] Final test/loss saved: {grid_path}")




# Save loss history
# loss_history.train_losses und val_losses in dict
loss_dict = {
    "train_losses": loss_history.train_losses,
    "val_losses": loss_history.val_losses
}

df = pd.DataFrame(loss_dict)
loss_path = os.path.join(run_dir, f"history_{DATA_NAME}_{name}.csv")
df.to_csv(loss_path, index=False)
print(f"[INFO] Final train history saved: {loss_path}")




# Plot training curves
# Werte
dataset_name = DATA_FILE
history_path = os.path.join(run_dir, f"history_{DATA_NAME}_{name}.csv")
history = pd.read_csv(history_path)
train_losses = history.train_losses
val_losses = history.val_losses

best_val_idx = val_losses.idxmin()
best_val_loss = val_losses.iloc[best_val_idx]
best_test_loss = test_metrics[0]["test/loss"]

# plt.figure(figsize=(8,5))

# Plot der Loss-Kurven
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Validation Loss", marker='s')
plt.plot([],[], linewidth=0, label=f"Test Loss: {best_test_loss:.4f}")
         
# Raster hinzufügen
plt.grid(True, linestyle='--', alpha=0.6)

# Beste Validation Loss markieren
plt.scatter(best_val_idx, best_val_loss, color='red', s=80, zorder=5, label="Best Val Loss")
plt.annotate(f"{best_val_loss:.4f}", (best_val_idx, best_val_loss),
             textcoords="offset points", xytext=(0,10), ha='center', color='red')

# Achsenbeschriftung
plt.xlabel("Epoch")
plt.ylabel("MSE-Loss")

# Titel nur mit Dataset und Hyperparameter
plt.title(
    f"NN with Dataset: {dataset_name}\n"
    f"HPs: L={L}, hidden dims={hidden_dims}, P={num_params}, batch size={batch_size}"
)

plt.legend()
plt.tight_layout()

# Speicherpfad
plot_path = os.path.join(run_dir, f"training_{DATA_NAME}_{name}.pdf")
plt.savefig(plot_path)
plt.show(block=False)

print(f"[INFO] Trainingsplot mit Markierungen gespeichert unter: {plot_path}")

