# gridsearch_regression_sdkn.py
# Diese script führt einen gridsearch mit einem SDKN
# auf den Datensätzen "dataset_2175_kin8nm.arff" oder "airfoil_self_noise.dat" aus.
# Vor Ausführung bitte sicher stellen:
# Die Klasse SDKN in utils/lightning_modules.py muss von "Network" erben.

import os
import json
import math
import itertools
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
    LossHistoryRegression,
    compute_centers
)




# Gridsearch Settings
target_params_list = [10000]
L_list = [1]

h1 = [80,90,100,110,120]
h2 = [30,40,50,60,70,80]
h3 = []

M_list = [10]
          

param_grid = {
    "P_target": target_params_list,
    "L": L_list,
    "h1": h1,
    #"h2": h2,
    #"h3": h3,
    "M": M_list
}

d0 = 8 # input dimension
d = 1 # output dimension

max_epochs = 50
batch_size = 12


DATA_FILE = "dataset_2175_kin8nm.arff" # "dataset_2175_kin8nm.arff", "airfoil_self_noise.dat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DATA_FILE.endswith(".dat"):
    DATA_NAME = "airfoil"
else:
    DATA_NAME = "kin8nm"



# Create unique run directory
timestamp = datetime.now().strftime("%d-%m_%H-%M")
run_dir = os.path.join("results", DATA_NAME, "gridsearch", f"{DATA_NAME}_sdkn_grid_{timestamp}")
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
    stratify=False
)

print("[INFO] Dataset sizes:")
print(f"  Train: {len(X_train_full)}")
print(f"  Val:   {len(X_val_full)}")
print(f"  Test:  {len(X_test_full)}\n")



# Helper: n(L,M) computation for fixed P
def compute_n(P_target, L, M, d0, d):
    a = L - 1
    b = d0 + d + M * L

    # Spezialfall: linear
    if a == 0:
        n_star = P_target / b
    else:
        n_star = (-b + math.sqrt(b*b + 4*a*P_target)) / (2*a)

    candidates = {
        max(1, math.floor(n_star)),
        max(1, math.ceil(n_star))
    }

    best_n = min(
        candidates,
        key=lambda n: abs(a*n*n + b*n - P_target)
    )

    return best_n




# Gridsearch
keys = list(param_grid.keys())
values = list(param_grid.values())

results = []

for combo in itertools.product(*values):
    # h* Indexe bestimmen
    hidden_dims = [combo[i] for i,k in enumerate(keys) if k.startswith("h")]

    params = {
        k: combo[i]
        for i, k in enumerate(keys)
        if not k.startswith("h")
    }
    if not hidden_dims:
        n = compute_n(params["P_target"], params["L"], params["M"], d0, d)
        hidden_dims = [n] * params["L"]
    params["hidden_dims"] = hidden_dims

    num_centers = params["M"]
    name = f"M={num_centers}_{hidden_dims}"
    combo_name = str(combo)
    combo_dir = os.path.join(run_dir,combo_name)
    os.makedirs(combo_dir, exist_ok=False)
    print(f"[INFO] Combo directory: {combo_dir}")
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

    # Random centers
    centers = compute_centers(train_loader_full, method="random", num_centers=params["M"])

    model = lightning_models.SDKN(
        centers=centers,
        L=params["L"],
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

    ckpt = os.path.join(combo_dir, "best.ckpt")
    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=1,
        dirpath=combo_dir, filename="best"
    )

    early_stop_final = EarlyStopping(monitor="val/loss", patience=10, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        callbacks=[early_stop_final, checkpoint_cb, loss_history],
    )

    trainer.fit(model, train_loader_full, val_loader)

    # Load best model
    best_final_model = lightning_models.SDKN.load_from_checkpoint(
        ckpt,
        centers=centers,
        L=params["L"],
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
    torch.save(best_final_model.state_dict(), os.path.join(combo_dir, "final_model.pt"))


    # --- Hyperparameter speichern ---
    hparams_path = os.path.join(combo_dir, "model_hparams.json")
    hparams = {
        "Datensatz": DATA_FILE,
        "P": num_params,
        "L": params["L"],
        "M": params["M"],
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

    print(f"[INFO] Modell + Hyperparameter gespeichert unter {combo_dir}")



    # Save test_loss
    test_loss = {
                "Datensatz": DATA_FILE,
                "P": num_params,
                "L": params["L"],
                "M": params["M"],
                "hidden_dims": hidden_dims,
                "test_loss": test_metrics[0]["test/loss"],
                "max_epochs": max_epochs,
                "batch_size": batch_size,
            }
    df = pd.DataFrame([test_loss])
    grid_path = os.path.join(combo_dir, f"test_loss_{DATA_NAME}_{name}.csv")
    df.to_csv(grid_path, index=False)
    print(f"[INFO] Final test/loss saved: {grid_path}")




    # Save loss history
    # loss_history.train_losses und val_losses in dict
    loss_dict = {
        "train_losses": loss_history.train_losses,
        "val_losses": loss_history.val_losses
    }

    df = pd.DataFrame(loss_dict)
    loss_path = os.path.join(combo_dir, f"history_{DATA_NAME}_{name}.csv")
    df.to_csv(loss_path, index=False)
    print(f"[INFO] Final train history saved: {loss_path}")




    # Plot training curves
    # Werte
    dataset_name = DATA_FILE
    history_path = os.path.join(combo_dir, f"history_{DATA_NAME}_{name}.csv")
    history = pd.read_csv(history_path)
    train_losses = history.train_losses
    val_losses = history.val_losses

    best_val_idx = val_losses.idxmin()
    best_val_loss = val_losses.iloc[best_val_idx]
    best_test_loss = test_metrics[0]["test/loss"]

    plt.figure(figsize=(8,5))

    # Plot der Loss-Kurven
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot([],[], linewidth=0, label=f"Test Loss: {best_test_loss:.4f}")
            
    # Raster hinzufügen
    plt.grid(True, linestyle='--', alpha=0.6)

    # Beste Validation Loss markieren
    plt.scatter(best_val_idx, best_val_loss, color='red', s=20, zorder=5, label="Best Val Loss")
    plt.annotate(f"{best_val_loss:.4f}", (best_val_idx, best_val_loss),
                textcoords="offset points", xytext=(0,10), ha='center', color='red')

    # Achsenbeschriftung
    plt.xlabel("Epochs")
    plt.ylabel("MSE-Loss")

    # Titel nur mit Dataset und Hyperparameter
    depth = params["L"]
    plt.title(
        f"SDKN with Dataset: {dataset_name}\n"
        f"HPs: L={depth}, M={num_centers}, hidden dims={hidden_dims}, P={num_params}, batch size={batch_size}"
    )

    plt.legend()
    plt.tight_layout()

    # Speicherpfad
    plot_path = os.path.join(combo_dir, f"training_{DATA_NAME}_{name}.pdf")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show(block=False)
    plt.close()

    print(f"[INFO] Trainingsplot mit Markierungen gespeichert unter: {plot_path}")

    results.append(
        {
            "Datensatz": DATA_FILE,
            "P_target": params["P_target"],
            "P": num_params,
            "L": params["L"],
            "M": params["M"],
            "hidden_dims": hidden_dims,
            "min_val_loss": best_val_loss,
            "test_loss": best_test_loss,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
        }
    )


# Save grid results
df = pd.DataFrame(results)
grid_path = os.path.join(run_dir, f"gridsearch_{DATA_NAME}_sdkn.csv")
df.to_csv(grid_path, index=False)
print(f"[INFO] Gridsearch saved: {grid_path}")
