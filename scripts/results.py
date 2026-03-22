import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# set paths
base_path = "/Users/lijiachen/PyCharmMiscProject/Machine Translation"
log_dir   = os.path.join(base_path, "logs")
out_dir   = os.path.join(base_path, "results")

os.makedirs(out_dir, exist_ok=True)

# parse log files
def parse_dropouts(path):
    basename = os.path.basename(path)
    name = re.sub(r"^log_", "", basename)
    name = re.sub(r"\.txt$", "", name)
    return name

def parse_logs(path):
    rows = []
    test_ppl = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("epoch"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            epoch_str, train_str, valid_str = parts[0], parts[1], parts[2]
            if epoch_str.lower() == "test":
                test_ppl = float(valid_str)
            else:
                rows.append({
                    "epoch":     int(epoch_str),
                    "train_ppl": float(train_str) if train_str != "-" else None,
                    "valid_ppl": float(valid_str),
                })
    return pd.DataFrame(rows), test_ppl

# load logs
log_paths = sorted(
    glob.glob(os.path.join(log_dir, "log_*.txt")),
    key=lambda p: float(parse_dropouts(p)),
)

if not log_paths:
    raise FileNotFoundError(f"No log_*.txt files found in '{log_dir}'")

data = {}
for path in log_paths:
    label = parse_dropouts(path)
    df, test_ppl = parse_logs(path)
    data[label] = (df, test_ppl)

dropout_labels = list(data.keys())

# build dataframes
valid_df = pd.DataFrame()
test_row = {}

for label, (df, test_ppl) in data.items():
    valid_df[f"Dropout {label}"] = df.set_index("epoch")["valid_ppl"]
    test_row[f"Dropout {label}"] = test_ppl

epochs    = valid_df.index.tolist()
col_names = list(valid_df.columns)

# print and save the perplexity table
table_rows = []
for ep in epochs:
    row = {"Valid. Perplexity": f"Epoch {ep}"}
    for col in col_names:
        val = valid_df.loc[ep, col]
        row[col] = f"{val:.2f}" if pd.notna(val) else "-"
    table_rows.append(row)

end_row = {"Valid. Perplexity": "End of training (test)"}
for col in col_names:
    val = test_row.get(col)
    end_row[col] = f"{val:.2f}" if val is not None else "-"
table_rows.append(end_row)

table_df = pd.DataFrame(table_rows).set_index("Valid. Perplexity")

print("Valid Perplexity per Epoch & Test Perplexity")
print(table_df.reset_index().to_string(index=False))
print()

csv_path = os.path.join(out_dir, "valid_perplexity_table.csv")
table_df.to_csv(csv_path)

# create a perplexity chart for each model
TRAIN_COLOR = "#e63946"
VALID_COLOR = "#2a9d8f"

for label, (df, test_ppl) in data.items():
    df_plot = df.set_index("epoch")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#0f0f14")
    ax.set_facecolor("#16161d")

    # train_ppl
    if df_plot["train_ppl"].notna().any():
        ax.plot(
            df_plot.index,
            df_plot["train_ppl"],
            label="Train PPL",
            color=TRAIN_COLOR,
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=3.5,
            markevery=5,
        )

    # valid_ppl
    ax.plot(
        df_plot.index,
        df_plot["valid_ppl"],
        label="Valid PPL",
        color=VALID_COLOR,
        linewidth=2.0,
        linestyle="-",
        marker="s",
        markersize=3.5,
        markevery=5,
    )

    # test_ppl as horizontal reference line
    if test_ppl is not None:
        ax.axhline(
            y=test_ppl,
            color=VALID_COLOR,
            linewidth=1.2,
            linestyle=":",
            alpha=0.7,
            label=f"Test PPL = {test_ppl:.2f}",
        )

    ax.set_title(
        f"Perplexity — Dropout {label}",
        color="#f0f0f0", fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Epoch", color="#aaaaaa", fontsize=11)
    ax.set_ylabel("Perplexity", color="#aaaaaa", fontsize=11)

    ax.tick_params(colors="#888888", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(which="major", color="#2a2a3a", linewidth=0.8, linestyle="--")
    ax.grid(which="minor", color="#1e1e2a", linewidth=0.4, linestyle=":")

    ax.legend(
        frameon=True, framealpha=0.3,
        facecolor="#1a1a24", edgecolor="#444455",
        labelcolor="#dddddd", fontsize=9, loc="upper right",
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"ppl_dropout_{label}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

print("\nTable and charts all saved.")