# lightning_models.py
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from utils import kernels, utilities, settings
from torchmetrics import AUROC
import os


class Network(pl.LightningModule):
    def __init__(self, roc_save_path=None,M=None,hidden_dims=None):
        super().__init__()
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("test/loss", loss, prog_bar=True)
        return loss

class BinaryClassificationNetwork(pl.LightningModule):
    def __init__(self, roc_save_path=None,M=None,hidden_dims=None):
        super().__init__()
        self.roc_save_path = roc_save_path
        if M == None:
            self.name = f"{hidden_dims}"
        else:
            self.name = f"M={M}_{hidden_dims}"

        # Metrics
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

        # 🔥 ROC storage
        self.val_probs = []
        self.val_targets = []

        self.test_probs = []
        self.test_targets = []

        self.best_threshold = 0.5

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_start(self):
        self.val_probs = []
        self.val_targets = []
        self.val_auc.reset()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        # CPU speichern (wichtig!)
        self.val_probs.append(probs.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.val_auc.update(probs.detach(), y.int())

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_probs) == 0:
            print("missing validation")
            return

        probs = torch.cat(self.val_probs).view(-1, 1)
        targets = torch.cat(self.val_targets).view(-1, 1).float()

        thresholds = torch.linspace(0, 1, 200, device=probs.device).view(1, -1)

        preds = (probs >= thresholds).int()

        tp = ((preds == 1) & (targets == 1)).sum(dim=0)
        fp = ((preds == 1) & (targets == 0)).sum(dim=0)
        fn = ((preds == 0) & (targets == 1)).sum(dim=0)
        tn = ((preds == 0) & (targets == 0)).sum(dim=0)

        acc = (tp + tn).float() / (tp + tn + fp + fn).clamp(min=1)
        best_idx = torch.argmax(acc)
        best_acc = acc[best_idx].item()
        self.best_threshold = thresholds[0, best_idx].item()

        # logs
        self.log("val/auc", self.val_auc.compute())
        self.log("val/acc", best_acc)
        self.log("val/best_threshold", self.best_threshold, prog_bar=True)

        # reset
        self.val_auc.reset()
        self.val_probs.clear()
        self.val_targets.clear()

    # --------------------
    # TEST
    # --------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        # 🔥 ROC Daten sammeln
        self.test_probs.append(probs.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        self.test_auc.update(probs, y.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        if len(self.test_probs) == 0:
            return

        probs = torch.cat(self.test_probs).view(-1, 1)
        targets = torch.cat(self.test_targets).view(-1, 1).float()

        if self.roc_save_path is not None:
            os.makedirs(self.roc_save_path, exist_ok=True)
            torch.save(
                {
                    "probs": probs,
                    "targets": targets
                },
                os.path.join(self.roc_save_path, f"test_roc_{self.name}.pt")
            )

        thresholds = torch.linspace(0, 1, 200, device=probs.device).view(1, -1)

        preds = (probs >= thresholds).int()

        tp = ((preds == 1) & (targets == 1)).sum(dim=0)
        fp = ((preds == 1) & (targets == 0)).sum(dim=0)
        fn = ((preds == 0) & (targets == 1)).sum(dim=0)
        tn = ((preds == 0) & (targets == 0)).sum(dim=0)

        acc = (tp + tn).float() / (tp + tn + fp + fn).clamp(min=1)
        best_idx = torch.argmax(acc)
        best_acc = acc[best_idx].item()
        self.best_threshold = thresholds[0, best_idx].item()

        # logs
        self.log("test/auc", self.test_auc.compute())
        self.log("test/acc", best_acc)
        self.log("test/best_threshold", self.best_threshold, prog_bar=True)

        self.test_auc.reset()
        self.test_probs.clear()
        self.test_targets.clear()
# ------------------------
# SDKN
# ------------------------
# erbt von BinaryClassificationNetwork oder Network
class SDKN(BinaryClassificationNetwork):
    def __init__(self, centers, L=4, hidden_dims=None, d0=5, d=1, kernel=None, learning_rate=1e-3,
                 mean_x=None, std_x=None, mean_y=None, std_y=None, roc_save_path=None):
        super().__init__(roc_save_path=roc_save_path, M=centers.shape[0],hidden_dims=hidden_dims)
        self.save_hyperparameters(ignore=["mean_x", "std_x", "mean_y", "std_y"])
        self.learning_rate = learning_rate
        

        # Normalisierung
        if mean_x is not None and std_x is not None:
            self.register_buffer("input_mean_x", mean_x)
            self.register_buffer("input_std_x", std_x)
        if mean_y is not None and std_y is not None:
            self.register_buffer("input_mean_y", mean_y)
            self.register_buffer("input_std_y", std_y)
        print()
        
        # Centers
        self.register_buffer("centers", centers)
        self.M = centers.shape[0]

        # Kernel
        if kernel is None:
            kernel = kernels.Gaussian()


        # dims = [d0, h1, h2, ..., hL]
        dims = [d0] + hidden_dims

        # => fcs[0] : d0 → h1
        #    fcs[1] : h1 → h2
        #    ...
        #    fcs[L-1] : h_{L-1} → h_L
        #    fcs[L]   : h_L → d   (Output-Layer)
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1], bias=False) for i in range(L)] +
            [nn.Linear(dims[-1], d, bias=False)]
        )

        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight)

        # L Aktivierungslayer 
        self.activs = nn.ModuleList([
            utilities.ActivFunc(hidden_dims[i], self.M, kernel=kernel)
            for i in range(L)
        ])

    def forward(self, x):
        centers = self.centers

        # L mal: FC_i + Activation_i
        for i in range(len(self.activs)):
            x = self.fcs[i](x)
            centers = self.fcs[i](centers)
            x, centers = self.activs[i](x, centers)

        # letztes FC (ohne Aktivierung)
        x = self.fcs[-1](x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=settings.decayEpochs_sdkn, gamma=settings.decayRate
        )
        return [optimizer], [scheduler]




# erbt von BinaryClassificationNetwork oder Network
class NN(BinaryClassificationNetwork):
    def __init__(self, L=4, hidden_dims=[32, 64, 48, 24], d0=5, d=1, learning_rate=1e-3,
                 mean_x=None, std_x=None, mean_y=None, std_y=None, roc_save_path=None):
        super().__init__(roc_save_path=roc_save_path,hidden_dims=hidden_dims)
        self.save_hyperparameters(ignore=["mean_x", "std_x", "mean_y", "std_y"])
        self.learning_rate = learning_rate

        # Input / Output normalization
        if mean_x is not None and std_x is not None:
            self.register_buffer("input_mean_x", mean_x)
            self.register_buffer("input_std_x", std_x)
        if mean_y is not None and std_y is not None:
            self.register_buffer("input_mean_y", mean_y)
            self.register_buffer("input_std_y", std_y)

        # Dynamische fully connected Layers
        dims = [d0] + hidden_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=True) for i in range(L)])
        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight)

        # Output Layer
        self.fc_out = nn.Linear(hidden_dims[-1], d, bias=True)
        nn.init.kaiming_normal_(self.fc_out.weight)

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x))
        x = self.fc_out(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=settings.decayEpochs_nn, gamma=settings.decayRate
        )
        return [optimizer], [scheduler]
    
