# Script um training histories zu plotten
# Dateien vom Format .csv im ordner "train_historys" werden gelesen und geplottet
# Der Plot wird in "train_historys" gespeichert
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

run_dir = "train_historys"


# Namen der Dateien, Namen der plots
files = ["higgs_M=5_[400, 400]","higgs_M=5_[1200, 1200, 1200, 1200, 1200, 1200]"]
names = ["2x400","6x1200"]
#names = ["SDKN", "NN"]

# Metriken, die geplottet werden sollen
trainloss = True
valloss = True
roc = True
acc = True

# Labels und Titel der figure
ylabel = "Loss, Accuracy, AUROC"
xlable = "Epochs"
title = "SDKN M=5 on the Higgs dataset "

# Anzahl der Spalten in der Legende
ncol = 2

plt.figure()


# Schleife, in der alle gewünschten Metriken eines Modells geplottet werden
for i, file in enumerate(files):
    grid_path = os.path.join(run_dir, f"history_{file}.csv")
    history = pd.read_csv(grid_path)

    grid_path = os.path.join(run_dir, f"test_loss_{file}.csv")
    test_loss = pd.read_csv(grid_path)

    plt.grid(True, linestyle='--', alpha=0.6)

    train_losses = history.train_losses
    epochs = np.arange(1, len(train_losses) + 1)


    # TRAIN LOSS
    if trainloss:
        line = plt.plot(
            epochs,
            train_losses,
            label=f"{names[i]} Train Loss",
            linestyle='-'
        )[0]

        color = line.get_color()


    # VALIDATION LOSS
    if valloss:
        val_losses = history.val_losses
        best_val_loss_idx = val_losses.idxmin()
        best_val_loss = val_losses.iloc[best_val_loss_idx]

        line = plt.plot(
            epochs,
            val_losses,
            color=color,
            label=f"{names[i]} Val Loss",
            linestyle='--'
        )[0]

        color = line.get_color()

        plt.scatter(
            best_val_loss_idx + 1,
            best_val_loss,
            s=20,
            zorder=5,
            color=color,
            marker='D',
            label=f"{names[i]} MVL: {best_val_loss:.4f}"
        )


    # ROC
    if roc:
        val_aucs = history.val_aucs
        best_val_auc_idx = val_aucs.idxmax()
        best_val_auc = val_aucs.iloc[best_val_auc_idx]

        line = plt.plot(
            epochs,
            val_aucs,
            color=color,
            label=f"{names[i]} Val AUROC",
            linestyle='-.'
        )[0]

        color = line.get_color()

        plt.scatter(
            best_val_auc_idx + 1,
            best_val_auc,
            s=20,
            zorder=5,
            marker='o',
            color=color,
            label=f"{names[i]} Max Val AUROC: {best_val_auc:.4f}"
        )


    # ACCURACY
    if acc:
        val_accs = history.val_accs
        best_val_acc_idx = val_accs.idxmax()
        best_val_acc = val_accs.iloc[best_val_acc_idx]

        line = plt.plot(
            epochs,
            val_accs,
            color=color,
            label=f"{names[i]} Val Accuracy",
            linestyle=':'
        )[0]

        color = line.get_color()

        plt.scatter(
            best_val_acc_idx + 1,
            best_val_acc,
            s=20,
            zorder=5,
            marker='x',
            color=color,
            label=f"{names[i]} Max Val Acc: {best_val_acc:.4f}"
        )


    # TEST LOSS (legend only)
    best_test_loss = test_loss.test_loss.iloc[0]

    plt.plot(
        [],
        [],
        linewidth=0,
        label=f"{names[i]} TL: {best_test_loss:.4f}"
    )

# Achsenbeschriftung
plt.xlabel(xlable)
plt.ylabel(ylabel)
plt.legend(ncol=ncol, bbox_to_anchor=(0.5, -0.19), loc="upper center")
plt.title(title)
plt.tight_layout()
plt.subplots_adjust(bottom=0.5)  # 🔥 wichtigplt.grid()

# Plot-Name aus allen Namen bauen

counter = 1

while True:
    plot_name = "train_history_plot" + "_".join(files) + f"{counter}" + ".pdf"
    plot_path = os.path.join(run_dir, plot_name)


    if not os.path.exists(plot_path):
        break

    counter += 1

# SAVE
plt.savefig(plot_path, bbox_inches="tight")
plt.show(block=False)

print(f"[INFO] saved to: {plot_path}")