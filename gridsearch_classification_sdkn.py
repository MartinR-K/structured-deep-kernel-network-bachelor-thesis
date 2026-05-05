# gridsearch_classification_sdkn.py
# Diese script führt einen gridsearch mit einem SDKN
# auf den Higgs Datensatz aus.
# Vor Ausführung bitte sicher stellen:
# Die Klasse SDKN in utils/lightning_modules.py muss von "BinaryClassificationNetwork" erben.

import os
import sys
import json
import math
import itertools
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import torch
torch.set_float32_matmul_precision("medium")  # Tensor Cores

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
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



# Create unique run directory
timestamp = datetime.now().strftime("%d-%m_%H-%M")
run_dir = os.path.join("results", "Higgs", "gridsearch", f"higgs_sdkn_grid_{timestamp}")
os.makedirs(run_dir, exist_ok=False)
print(f"[INFO] Run directory: {run_dir}")



# Gridsearch Settings
GRID_ROWS = 1_000_000  # zufälliges Subset für Gridsearch
DATA_FILE = "HIGGS.csv"

target_params_list = [50000]
L_list = [4]

h1 = [400]
h2 = [400]
h3 = [400]
h4 = [400]
h5 = []
h6 = []

M_list = [5,10,20]

param_grid = {
    "P_target": target_params_list,
    "L": L_list,
    "h1": h1,
    "h2": h2,
    "h3": h3,
    "h4": h4,
    #"h5": h5,
    #"h6": h6,
    "M": M_list
}

d0 = 28 # input dimension
d = 1 # output dimension

max_epochs = 10
batch_size = 4096
k = 3 # CV folds (minimum 2)
num_workers = 8

num_epochs_final_training = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




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


# Subsample for Gridsearch
print(f"[INFO] Using subset of {GRID_ROWS} samples for gridsearch")
indices = torch.randperm(len(X_train_full))[:GRID_ROWS]
X_grid = X_train_full[indices]
y_grid = y_train_full[indices]



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
        n = compute_n(params["P_target"], params["L"], params["M"], d0, d)
        hidden_dims = [n] * params["L"]
    params["hidden_dims"] = hidden_dims


#for P_target, L, M in itertools.product(target_params_list, L_list, M_list):
    #a = L - 1
    #b = d0 + d + M * L
    #c_quad = -P_target
    #h = int(max(1, round((-b + math.sqrt(b**2 - 4*a*c_quad)) / (2*a))))

    fold_losses = []
    fold_aurocs = []
    fold_acc = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_grid)):
        print(f"[INFO] Fold {fold_idx + 1}/{k}")
        print(f"\n[INFO] Grid Param: P_target={params['P_target']}, L={params['L']}, M={params['M']}, hidden_dims={hidden_dims}")

        X_train, y_train = X_grid[train_idx], y_grid[train_idx]
        X_val, y_val     = X_grid[val_idx], y_grid[val_idx]

        X_train, y_train, X_val, y_val, mean_x, std_x, _, _ = normalize_train_only(
            X_train, y_train, (X_val, y_val), only_features=True
        )

        train_loader = make_loader(X_train, y_train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader   = make_loader(X_val, y_val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        centers = compute_centers(train_loader, method="random", num_centers=params["M"])

        model = lightning_models.SDKN(
            centers=centers, 
            L=params["L"],
            hidden_dims=hidden_dims,
            d0=d0, 
            d=d, 
            mean_x=mean_x, 
            std_x=std_x,
            roc_save_path=run_dir,
        ).to(DEVICE)

        # Early stopping + checkpoint inside fold
        fold_ckpt = os.path.join(run_dir, f"fold_{fold_idx}_best.ckpt")
        checkpoint_cb = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1, # = 1 für speichern des besten models
            dirpath=run_dir,
            filename=f"fold_{fold_idx}_best"
        )
        early_stop = EarlyStopping(monitor='val/loss', patience=5, mode='min')

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=1,
            precision=16,
            callbacks=[early_stop, checkpoint_cb],
            num_sanity_val_steps=0,
            logger=False
        )

        trainer.fit(model, train_loader, val_loader)


        val_metrics = trainer.validate(ckpt_path="best", dataloaders=val_loader)
        #val_loss = trainer.callback_metrics.get("val/loss")
        #val_auc  = trainer.callback_metrics.get("val/auc")
        #val_loss = float(val_loss) if val_loss is not None else float("inf")
        #val_auc  = float(val_auc) if val_auc is not None else 0.0

        fold_losses.append(val_metrics[0]["val/loss"])
        fold_aurocs.append(val_metrics[0]["val/auc"])

    mean_cv_loss = float(sum(fold_losses) / len(fold_losses))
    mean_auroc = float(sum(fold_aurocs) / len(fold_aurocs))
    print(f"[INFO] mean CV loss = {mean_cv_loss:.6f}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    results.append(
        {
            "Datensatz": DATA_FILE, 
            "P_target": params["P_target"],
            "P": num_params,
            "L": params["L"],
            "M": params["M"],
            "hidden_dims": hidden_dims,
            "cv_loss": mean_cv_loss, 
            "cv_auroc": mean_auroc,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "CV #folds": k,
        }
    )




# Save grid results
df = pd.DataFrame(results)
grid_path = os.path.join(run_dir, "gridsearch_higgs_sdkn.csv")
df.to_csv(grid_path, index=False)
print(f"[INFO] Gridsearch saved: {grid_path}")


# Select best tuple (highest AUROC)
best_row = df.loc[[df["cv_auroc"].idxmax()]]
best_row_path = os.path.join(run_dir, "highest_auroc_gridsearch_higgs.csv")
best_row.to_csv(best_row_path, index=False)

P_best_target, P_best, L_best, M_best, hidden_dims_best = (
    int(best_row["P_target"]),
    int(best_row["P"]),
    int(best_row["L"]),
    int(best_row["M"]),
    best_row["hidden_dims"],
)

print(f"\n[INFO] Best hyperparameters: {best_row.to_dict()}\n")

