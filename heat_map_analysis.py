# Script um den mittleren loss des oberen linken und des unteren rechten
# Dreiecks von heatmaps zu berechnen
# files werden aus dem "heat_maps" gelesen und plots werden dort gespeichert
import os
import pandas as pd
import ast



# EINSTELLUNGEN

run_dir = "heat_maps"

# True  -> single csv with split parameter
# False -> multiple csv files
use_single_file_mode = True

# MULTI FILE MODE
csv_files = [
    "gridsearch_cv_sdkn (copy).csv",
    "gridsearch_cv_sdkn (another copy).csv",
    "gridsearch_cv_sdkn (3rd copy).csv",
    "gridsearch_cv_sdkn (4th copy).csv",
    "gridsearch_cv_sdkn (5th copy).csv",
]

titles = [
    "M=5",
    "M=10",
    "M=20",
    "M=30",
    "M=40",
]

# SINGLE FILE MODE
csv_file = "gridsearch_airfoil_sdkn.csv"
split_param = "M"
split_values = None   # automatic if None



metric = "cv_loss"

array_param = "hidden_dims"
array_index_x = 0   # h1
array_index_y = 1   # h2

output_csv = f"triangle_loss_{csv_file}" # .csv is included

# HELPERS

def prepare_df(df):
    df = df.copy()

    df[array_param] = df[array_param].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df["h1"] = df[array_param].apply(lambda x: x[array_index_x])
    df["h2"] = df[array_param].apply(lambda x: x[array_index_y])

    return df


def triangle_means(df):
    upper_left = df[df["h2"] > df["h1"]]
    lower_right = df[df["h1"] > df["h2"]]
    diagonal = df[df["h1"] == df["h2"]]

    upper_mean = upper_left[metric].mean()
    lower_mean = lower_right[metric].mean()
    diag_mean = diagonal[metric].mean()

    return {
        "upper_left_mean": upper_mean,
        "lower_right_mean": lower_mean,
        "diagonal_mean": diag_mean,
        "difference_ul_minus_lr": upper_mean - lower_mean,
        "n_upper_left": len(upper_left),
        "n_lower_right": len(lower_right),
        "n_diagonal": len(diagonal)
    }


# ANALYSIS

results = []

if use_single_file_mode:
    df_all = pd.read_csv(os.path.join(run_dir, csv_file))
    df_all = prepare_df(df_all)

    if split_values is None:
        split_values = sorted(df_all[split_param].unique())

    for split_value in split_values:
        df_subset = df_all[df_all[split_param] == split_value]

        result = triangle_means(df_subset)

        result[split_param] = split_value

        results.append(result)

else:
    for file, title in zip(csv_files, titles):
        df = pd.read_csv(os.path.join(run_dir, file))
        df = prepare_df(df)

        result = triangle_means(df)

        # optional: M-Wert aus Titel extrahieren
        try:
            m_value = int(title.split("=")[-1])
        except:
            m_value = title

        result["M"] = m_value
        result["source_file"] = file

        results.append(result)

# SAVE CSV

results_df = pd.DataFrame(results)

# schöne Spaltenreihenfolge
preferred_cols = [
    split_param if use_single_file_mode else "M",
    "upper_left_mean",
    "lower_right_mean",
    "diagonal_mean",
    "difference_ul_minus_lr",
    "n_upper_left",
    "n_lower_right",
    "n_diagonal"
]

# source_file nur falls vorhanden
if "source_file" in results_df.columns:
    preferred_cols.append("source_file")

results_df = results_df[preferred_cols]

output_path = os.path.join(run_dir, output_csv)
results_df.to_csv(output_path, index=False)

print(f"[INFO] saved results to: {output_path}")
print(results_df)
