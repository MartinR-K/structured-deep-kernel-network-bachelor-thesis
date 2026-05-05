# Script zum plotten von heat maps
# Dateien vom Format .csv im ordner "heat_maps" werden gelesen und geplottet
# Der Plot wird in "heat_maps" gespeichert
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Textgrößen 
plt.rcParams.update({
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
    "figure.titlesize": 30
})



# DATEI EINSTELLUNGEN

run_dir = "heat_maps"

# MODE SWITCH
use_single_file_mode = False
use_shared_colorbar = False

# MULTI-FILE MODE
csv_files = [
    "gridsearch_kin8nm_nn.csv",
    "gridsearch_kin8nm_sdkn.csv"
]

titles = [
    "NN on kin8nm",
    "SDKN on kin8nm"
]

# SINGLE-FILE MODE
csv_file = "gridsearch_kin8nm_nn.csv"

MODEL = "nn"

split_param = "batch_size"

if csv_file.startswith("gridsearch_air"):
    DATA_NAME = "airfoil"
else:
    DATA_NAME = "kin8nm"

split_values = None

# PLOT EINSTELLUNGEN
# EXTREMA DISPLAY MODE
# "text"    -> values directly inside heatmap
# "markers" -> markers + legend below plot
extrema_display_mode = "markers"

# Y-AXIS LABEL MODE  (muss 'first', 'all', oder 'none' sein)
y_axis_label_mode = "all"

# Y-AXIS TICK MODE (muss 'first', 'all', oder 'none' sein)
y_tick_mode = "all"

metric = "cv_loss"
metric_name = "cv loss"

use_2d_params = False

x_param = "L"
y_param = "P_target"
x_param_label = "numbers of hidden layers"
y_param_label = "parameters"

array_param = "hidden_dims"
array_param_label = ["number of hidden layers", "parameters"]

array_index_x = 0
array_index_y = 1

cmap = "Greys_r"
text_color_min = "white"
text_color_max = "black"

# HELPERS

def prepare_df(df):
    df = df.copy()

    if use_2d_params:
        df[array_param] = df[array_param].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        df["plot_x"] = df[array_param].apply(
            lambda x: x[array_index_x]
        )

        df["plot_y"] = df[array_param].apply(
            lambda x: x[array_index_y]
        )

    else:
        df["plot_x"] = df[x_param]
        df["plot_y"] = df[y_param]

    return df


def get_axis_labels():
    if use_2d_params:
        return array_param_label[0], array_param_label[1]
    return x_param_label, y_param_label


def should_show_ylabel(plot_idx):
    if y_axis_label_mode == "first":
        return plot_idx == 0
    elif y_axis_label_mode == "all":
        return True
    elif y_axis_label_mode == "none":
        return False
    else:
        raise ValueError(
            "y_axis_label_mode must be 'first', 'all', or 'none'"
        )


def should_show_yticks(plot_idx):
    if y_tick_mode == "first":
        return plot_idx == 0
    elif y_tick_mode == "all":
        return True
    elif y_tick_mode == "none":
        return False
    else:
        raise ValueError(
            "y_tick_mode must be 'first', 'all', or 'none'"
        )


def create_heatmap(
    ax,
    df,
    title,
    vmin,
    vmax,
    show_ylabel,
    show_yticks
):
    heatmap_data = df.pivot_table(
        index="plot_y",
        columns="plot_x",
        values=metric,
        aggfunc="mean"
    )

    img = ax.imshow(
        heatmap_data,
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower"
    )

    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)

    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    ax.set_xlabel(x_label)

    if show_ylabel:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    if not show_yticks:
        ax.tick_params(axis="y", labelleft=False, left=False)

    ax.set_title(title)

    # extrema
    min_idx = np.unravel_index(
        np.nanargmin(heatmap_data.values),
        heatmap_data.shape
    )

    max_idx = np.unravel_index(
        np.nanargmax(heatmap_data.values),
        heatmap_data.shape
    )

    min_row, min_col = min_idx
    max_row, max_col = max_idx

    min_value = heatmap_data.iloc[min_row, min_col]
    max_value = heatmap_data.iloc[max_row, max_col]

    if extrema_display_mode == "text":
        ax.text(
            min_col,
            min_row,
            f"{min_value:.2f}",
            ha="center",
            va="center",
            color=text_color_min
        )

        ax.text(
            max_col,
            max_row,
            f"{max_value:.2f}",
            ha="center",
            va="center",
            color=text_color_max
        )

    elif extrema_display_mode == "markers":
        ax.plot(
            min_col,
            min_row,
            "o",
            markersize=12,
            markeredgewidth=2
        )

        ax.plot(
            max_col,
            max_row,
            "o",
            markersize=12,
            markeredgewidth=2,
        )

    extrema_info = {
        "title": title,
        "min_value": min_value,
        "max_value": max_value
    }

    return img, extrema_info




# LOAD DATA

x_label, y_label = get_axis_labels()

if use_single_file_mode:
    df_all = pd.read_csv(os.path.join(run_dir, csv_file))
    df_all = prepare_df(df_all)

    if split_values is None:
        split_values = sorted(df_all[split_param].unique())

    titles = [f"{split_param}={v}" for v in split_values]

    all_values = df_all[metric].dropna().tolist()
    n_plots = len(split_values)

else:
    all_values = []

    for file in csv_files:
        df = pd.read_csv(os.path.join(run_dir, file))
        df = prepare_df(df)
        all_values.extend(df[metric].dropna().tolist())

    n_plots = len(csv_files)

# COLOR SCALING

if use_shared_colorbar:
    global_vmin = min(all_values)
    global_vmax = max(all_values)
else:
    global_vmin = None
    global_vmax = None

# PLOT SETUP

fig, axes = plt.subplots(
    1,
    n_plots,
    figsize=(7 * n_plots, 6),
    squeeze=False
)

axes = axes[0]

subplot_imgs = []
extrema_infos = []



# CREATE HEATMAPS

if use_single_file_mode:
    for plot_idx, (ax, split_value, title) in enumerate(
        zip(axes, split_values, titles)
    ):
        df_subset = df_all[df_all[split_param] == split_value]

        if use_shared_colorbar:
            local_vmin = global_vmin
            local_vmax = global_vmax
        else:
            local_vmin = df_subset[metric].min()
            local_vmax = df_subset[metric].max()

        img, extrema_info = create_heatmap(
            ax=ax,
            df=df_subset,
            title="NN on kin8nm",
            vmin=local_vmin,
            vmax=local_vmax,
            show_ylabel=should_show_ylabel(plot_idx),
            show_yticks=should_show_yticks(plot_idx)
        )

        subplot_imgs.append(img)
        extrema_infos.append(extrema_info)

        if not use_shared_colorbar:
            fig.colorbar(
                img,
                ax=ax,
                label=f"MSE {metric_name}",
                shrink=0.8
            )

else:
    for plot_idx, (ax, file, title) in enumerate(
        zip(axes, csv_files, titles)
    ):
        df = pd.read_csv(os.path.join(run_dir, file))
        df = prepare_df(df)

        if use_shared_colorbar:
            local_vmin = global_vmin
            local_vmax = global_vmax
        else:
            local_vmin = df[metric].min()
            local_vmax = df[metric].max()

        img, extrema_info = create_heatmap(
            ax=ax,
            df=df,
            title=title,
            vmin=local_vmin,
            vmax=local_vmax,
            show_ylabel=should_show_ylabel(plot_idx),
            show_yticks=should_show_yticks(plot_idx)
        )

        subplot_imgs.append(img)
        extrema_infos.append(extrema_info)

        if not use_shared_colorbar:
            fig.colorbar(
                img,
                ax=ax,
                label=f"MSE {metric_name}",
                shrink=0.8
            )


# SHARED COLORBAR

plt.tight_layout()

if use_shared_colorbar:
    fig.subplots_adjust(right=0.88)

    fig.colorbar(
        subplot_imgs[-1],
        ax=axes,
        label=f"MSE {metric_name}",
        fraction=0.03,
        pad=0.02
    )

# MARKER LEGEND

if extrema_display_mode == "markers":
    for ax, info in zip(axes, extrema_infos):
        ax.text(
            0.5,
            -0.3,
            f"min={info['min_value']:.2f}, max={info['max_value']:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=30
        )

    plt.subplots_adjust(bottom=0.22)



# SAVE

counter = 1

while True:
    plot_name = f"{DATA_NAME}_{MODEL}_heatmap_{counter}.pdf"
    plot_path = os.path.join(run_dir, plot_name)

    if not os.path.exists(plot_path):
        break

    counter += 1

plt.savefig(plot_path, bbox_inches="tight")
plt.show(block=False)

print(f"[INFO] saved to: {plot_path}")