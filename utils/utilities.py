# utilities.py
import os
import numpy as np
import math
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from utils import kernels, settings

class TorchDataset(Dataset):
    def __init__(self, data_input, data_output):
        self.data_input = data_input
        self.data_output = data_output

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return (self.data_input[idx], self.data_output[idx])

class ActivFunc(torch.nn.Module):
    def __init__(self, in_features, nr_centers, kernel=None):
        super().__init__()
        self.in_features = in_features
        self.nr_centers = nr_centers
        self.nr_centers_id = nr_centers

        self.kernel = kernel if kernel else kernels.Wendland_order_0(ep=1)
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_features, self.nr_centers_id))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data += 0.2

    def forward(self, x, centers):
        cx = torch.cat((centers, x), 0)
        dist_matrix = torch.abs(cx.unsqueeze(2) - centers.t().view(1, centers.shape[1], self.nr_centers))
        kernel_matrix = self.kernel.rbf(self.kernel.ep, dist_matrix)
        cx = torch.sum(kernel_matrix * self.weight, dim=2)
        return cx[self.nr_centers:, :], cx[:self.nr_centers, :]

class LossHistoryRegression(pl.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train/loss")
        val_loss = trainer.callback_metrics.get("val/loss")

        # nur loggen wenn beide vorhanden sind
        if train_loss is None or val_loss is None:
            print(f"[WARN] missing loss in epoch {trainer.current_epoch}")
            return

        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))

class LossHistory(pl.callbacks.Callback):
    def __init__(self, roc_save_path=None):
        self.roc_save_path = roc_save_path

        self.train_losses = []
        self.val_losses = []
        self.val_auc = []
        self.val_best_threshold = []
        self.val_acc = []

        self.val_probs = []
        self.val_targets = []


    def on_validation_epoch_end(self, trainer, pl_module):
        
        train_loss = trainer.callback_metrics.get("train/loss")
        val_loss = trainer.callback_metrics.get("val/loss")
        auc = trainer.callback_metrics.get("val/auc")
        best_threshold = trainer.callback_metrics.get("val/best_threshold")
        acc = trainer.callback_metrics.get("val/acc")

        # 🔒 alles oder nichts damit alle listen die gleiche länge haben
        if None in (train_loss, val_loss, auc, best_threshold, acc):
            print(f"train/loss, val/loss, val/auc, val/best_threshold or val/acc is missing in {trainer.current_epoch}")
            print(f"train/loss={train_loss}")
            print(f"val/loss={val_loss}")
            print(f"auc={auc}")
            print(f"best_threshold={best_threshold}")
            print(f"val/acc={acc}")
            return

        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.val_auc.append(float(auc))
        self.val_best_threshold.append(float(best_threshold))
        self.val_acc.append(float(acc))

        if (hasattr(pl_module, "val_probs") and hasattr(pl_module, "val_targets") 
            and len(pl_module.val_probs) > 0 
            and len(pl_module.val_probs)==len(pl_module.val_targets)):
            probs = torch.cat(pl_module.val_probs).detach().cpu()
            targets = torch.cat(pl_module.val_targets).detach().cpu()

            self.val_probs.append(probs)
            self.val_targets.append(targets)

            # 🔥 speichern
            if self.roc_save_path is not None:
                save_dir = os.path.join(self.roc_save_path, "val_rocs")
                os.makedirs(save_dir, exist_ok=True)

                torch.save(
                    {
                        "probs": probs,
                        "targets": targets
                    },
                    os.path.join(save_dir, f"roc_epoch_{trainer.current_epoch}.pt")
                )


def train_test_split_dataset(X, y, test_size=0.15):
    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=test_size, shuffle=True, random_state=42
    )
    return X[idx_train], y[idx_train], X[idx_test], y[idx_test]


def train_val_test_split_dataset(X, y, test_size=0.1, val_size=0.1, seed=42, stratify=True):
    # First split: Test vs Rest
    strat = y if stratify else None
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    # Second split: Train vs Val
    strat_rest = y_rest if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=val_size / (1.0 - test_size), random_state=seed, stratify=strat_rest)
    return X_train, y_train, X_val, y_val, X_test, y_test

def normalize_train_only(X_train, y_train, *other_splits, only_features=False):
    """
    Normalisiert nur anhand des Trainingssatzes.
    Kann beliebig viele weitere Splits verarbeiten.

    Rückgabe:
        X_train_n, y_train_n, <alle anderen splits>, mean_x, std_x, mean_y, std_y
    """

    # --- Compute stats from TRAIN only ---
    mean_x = X_train.mean(dim=0)
    std_x = X_train.std(dim=0)
    std_x[std_x == 0] = 1.0

    if only_features==False:
        mean_y = y_train.mean(dim=0)
        std_y = y_train.std(dim=0)
        std_y[std_y == 0] = 1.0
    else:
        mean_y = None
        std_y = None

    # Normalize train
    X_train_n = (X_train - mean_x) / std_x
    if only_features==False:
        y_train_n = (y_train - mean_y) / std_y
    else:
        y_train_n = y_train

    # Process other splits
    normalized_splits = []
    for (X_sp, y_sp) in other_splits:
        X_sp_n = (X_sp - mean_x) / std_x
        if only_features==False:
            y_sp_n = (y_sp - mean_y) / std_y
        else:
            y_sp_n = y_sp
        normalized_splits.extend([X_sp_n, y_sp_n])

    return X_train_n, y_train_n, *normalized_splits, mean_x, std_x, mean_y, std_y
    

def make_loader(X, y, batch_size, shuffle=False, num_workers=7, pin_memory=True):
    return DataLoader(TorchDataset(X, y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

def compute_centers(train_loader, method="kmeans", num_centers=50):
    X = torch.cat([x for x, _ in train_loader], dim=0)

    if method in ["kmeans", "kmeans++"]:
        kmeans = KMeans(n_clusters=num_centers, init='k-means++' if method=="kmeans++" else 'random', random_state=0)
        kmeans.fit(X.cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=X.device)
    elif method == "random":
        indices = torch.randperm(X.size(0), device=X.device)[:num_centers]
        centers = X[indices]
    elif method == "uniform":
        min_vals, _ = X.min(dim=0)
        max_vals, _ = X.max(dim=0)
        centers = torch.rand(num_centers, X.size(1), device=X.device) * (max_vals - min_vals) + min_vals
    else:
        raise ValueError("Unknown center initialization method.")

    return centers

def load_full_dataset(my_data_file):
    file_path = os.path.join(settings.base_path, my_data_file)


    # ARFF FILES
    if my_data_file.endswith(".arff"):
        data, _ = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        arr = df.values.astype(np.float32)

        X = torch.tensor(arr[:, :-1])
        y = torch.tensor(arr[:, -1:], dtype=torch.float32)
        return X, y


    # CSV FILES
    elif my_data_file.endswith(".csv"):

        # CSV einlesen
        df = pd.read_csv(file_path)

        # --- HIGGS-FORMAT ERKENNEN ---
        # HIGGS: 29 Spalten -> label in col 0
        if df.shape[1] == 29:
            label_col = 0
        else:
            # Allgemeiner Fall: letzte Spalte = Label
            label_col = -1

        # Split
        if label_col == 0:
            y = df.iloc[:, 0].values.astype(np.float32)
            X = df.iloc[:, 1:].values.astype(np.float32)
        else:
            y = df.iloc[:, -1].values.astype(np.float32)
            X = df.iloc[:, :-1].values.astype(np.float32)

        # Torch Tensors
        X = torch.tensor(X)
        y = torch.tensor(y).reshape(-1, 1)
        return X, y



    # Fallback: Textdatei (z.B. .data, .txt)
    else:
        arr = np.loadtxt(file_path).astype(np.float32)
        X = torch.tensor(arr[:, :-1])
        y = torch.tensor(arr[:, -1:], dtype=torch.float32)
        return X, y


def get_DataLoader(my_data_file):

    "Feature normalization was performed using z-score normalization. "
    "The mean and standard deviation were computed exclusively from the training set to prevent data leakage."
    # --- Load file ---
    file_path = os.path.join(settings.base_path, my_data_file)
    if my_data_file.endswith(".arff"):
        data, _ = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        data_array = df.values.astype(np.float32)

    else:  # assume text file
        data_array = np.loadtxt(file_path).astype(np.float32)

    data_tensor = torch.tensor(data_array)

    # --- Automatic split inputs / targets ---
    n_cols = data_tensor.shape[1]

    inputs = data_tensor[:, :-1]   # all but last column
    labels = data_tensor[:, -1:].clone()  # last column

    # --- Create full dataset ---
    full_dataset = TorchDataset(inputs, labels)

    # --- Train / Val / Test split ---
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    # --- Compute normalization ONLY on training data ---
    train_inputs = torch.stack([dataset_train[i][0] for i in range(len(dataset_train))]) # dataset_train.shape = (len(dataset_train),2)
    train_labels = torch.stack([dataset_train[i][1] for i in range(len(dataset_train))])

    mean_x = train_inputs.mean(dim=0)
    std_x  = train_inputs.std(dim=0)
    std_x[std_x == 0] = 1.0   # avoid division by zero

    mean_y = train_labels.mean(dim=0)
    std_y  = train_labels.std(dim=0)
    std_y[std_y == 0] = 1.0

    # --- Apply normalization ---
    def normalize_dataset_input(dataset):
        normalized_data = []
        for x, y in dataset:
            x_norm = (x - mean_x) / std_x
            normalized_data.append((x_norm, y))
        return normalized_data
    
    def normalize_dataset(dataset):
        normalized_data = []
        for x, y in dataset:
            x_norm = (x - mean_x) / std_x
            y_norm = (y - mean_y) / std_y
            normalized_data.append((x_norm, y_norm))
        return normalized_data

    dataset_train = normalize_dataset(dataset_train)
    dataset_val   = normalize_dataset(dataset_val)
    dataset_test  = normalize_dataset(dataset_test)

    # --- Create DataLoaders ---
    train_loader = DataLoader(dataset_train,
                              batch_size=settings.batch_size,
                              shuffle=True,
                              num_workers=settings.num_workers,
                              pin_memory=True)

    val_loader = DataLoader(dataset_val,
                            batch_size=settings.batch_size,
                            num_workers=settings.num_workers,
                            pin_memory=True)

    test_loader = DataLoader(dataset_test,
                             batch_size=settings.batch_size,
                             num_workers=settings.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader, mean_x, std_x, mean_y, std_y
