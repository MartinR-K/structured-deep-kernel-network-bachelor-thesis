# gridsearch_regression_nn_cross_val.py
# Diese script führt einen gridsearch mit einem feedforward neural network mit ReLU activation 
# auf den Datensätzen "dataset_2175_kin8nm.arff" oder "airfoil_self_noise.dat" mit cross validation aus.
# Vor Ausführung bitte sicher stellen:
# Die Klasse NN in utils/lightning_modules.py muss von "Network" erben.

import sys
import os
import json
import math
import itertools
import pandas as pd
import numpy as np
import torch
torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from datetime import datetime

from utils import lightning_models
from utils.utilities import (
    load_full_dataset,
    train_val_test_split_dataset,
    normalize_train_only,
    make_loader
)




# Gridsearch Settings
target_params_list = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
L_list = [2,3,4,5,6]

h1 = [30,40,50,60,70,80]
h2 = [30,40,50,60,70,80]
h3 = []

param_grid = {
    "P_target": target_params_list,
    "L": L_list,
    #"h1": h1,
    #"h2": h2,
    #"h3": h3,
}

d0 = 8 # input dimension
d = 1 # output dimension

num_epochs = 15
batch_size=12
k = 3  # CV folds

num_epochs_final_training = 20

DATA_FILE = "dataset_2175_kin8nm.arff" # "dataset_2175_kin8nm.arff", "airfoil_self_noise.dat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DATA_FILE.endswith(".dat"):
    DATA_NAME = "airfoil"
else:
    DATA_NAME = "kin8nm"




# Create unique run directory
timestamp = datetime.now().strftime("%d-%m_%H-%M")
run_dir = os.path.join("results", DATA_NAME, "gridsearch", f"{DATA_NAME}_nn_grid_{timestamp}")
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





# Helper: n(L) computation for fixed P
def compute_n(P_target, L, d0, d):
    a = L - 1
    b = L + d + d0
    c = d

    # linearer Spezialfall: L = 1
    if a == 0:
        n_star = (P_target - c) / b
    else:
        disc = b*b + 4*a*(P_target - c)
        n_star = (-b + math.sqrt(disc)) / (2*a)

    candidates = {
        max(1, math.floor(n_star)),
        max(1, math.ceil(n_star))
    }

    best_n = min(
        candidates,
        key=lambda n: abs(a*n*n + b*n + c - P_target)
    )
    return best_n




# Gridsearch
kf = KFold(n_splits=k, shuffle=True)
results = []

keys = list(param_grid.keys())
values = list(param_grid.values())

for combo in itertools.product(*values):
    # h* Indexe bestimmen
    hidden_dims = [combo[i] for i,k in enumerate(keys) if k.startswith("h")]

    params = {
        k: combo[i]
        for i, k in enumerate(keys)
        if not k.startswith("h")
    }
    if not hidden_dims:
        n = compute_n(params["P_target"], params["L"], d0, d)
        hidden_dims = [n] * params["L"]
    params["hidden_dims"] = hidden_dims
    
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):

        print(f"\n[INFO]   Fold {fold_idx+1}/{k}")
        print(f"[INFO] Grid candidate: P_target={params['P_target']}, L={params['L']}, hidden_dims={hidden_dims}")

        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val_fold = X_train_full[val_idx]
        y_val_fold = y_train_full[val_idx]

        # Normalize (train statistics only)
        X_train_n, y_train_n, X_val_n, y_val_n, mean_x, std_x, mean_y, std_y = normalize_train_only(
            X_train, y_train, (X_val_fold, y_val_fold)
        )

        train_loader = make_loader(X_train_n, y_train_n, batch_size, shuffle=True)
        val_loader = make_loader(X_val_n, y_val_n, batch_size, shuffle=False)

        model = lightning_models.NN(
            L=params["L"],
            hidden_dims=hidden_dims,
            d0=d0,
            d=d,
            mean_x=mean_x,
            std_x=std_x,
            mean_y=mean_y,
            std_y=std_y,
        ).to(DEVICE)

        # Early stopping + checkpoint inside fold
        fold_ckpt = os.path.join(run_dir, f"fold_{fold_idx}_best.ckpt")
        checkpoint_cb = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1, # = 1 für speichern des besten models
            dirpath=run_dir,
            filename=f"fold_{fold_idx}_best",
        )

        early_stop = EarlyStopping(monitor="val/loss", patience=5, mode="min")

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="gpu" if DEVICE == "cuda" else "cpu",
            devices=1,
            logger=False,
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_cb],
        )

        trainer.fit(model, train_loader, val_loader)
        
        val_metrics = trainer.validate(ckpt_path="best", dataloaders=val_loader)
        fold_losses.append(val_metrics[0]["val/loss"])

    # mean CV loss
    mean_cv_loss = sum(fold_losses)/len(fold_losses)
    print(f"[INFO] mean CV loss: {mean_cv_loss}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results.append(
        {
            "Datensatz": DATA_FILE,
            "P_target": params["P_target"],
            "P": num_params,
            "L": params["L"],
            "hidden_dims": hidden_dims,
            "cv_loss": mean_cv_loss,
            "max_epochs": num_epochs,
            "batch_size": batch_size,
            "CV #folds": k,
        }
    )




# Save grid results
df = pd.DataFrame(results)
grid_path = os.path.join(run_dir, f"gridsearch_{DATA_NAME}_nn.csv")
df.to_csv(grid_path, index=False)
print(f"[INFO] Gridsearch saved: {grid_path}")




# Select best tuple
best_row = df.loc[df["cv_loss"].idxmin()]
best_row = df.loc[df["cv_loss"].idxmin()]
best_row_path = os.path.join(run_dir, f"lowest_valloss_gridsearch_{DATA_NAME}.csv")
best_row.to_csv(best_row_path, index=False)

P_best_target, P_best, L_best, hidden_dims_best = (
    int(best_row["P_target"]),
    int(best_row["P"]),
    int(best_row["L"]),
    best_row["hidden_dims"],
)

print(f"\n[INFO] Best hyperparameters: {best_row.to_dict()}\n")
