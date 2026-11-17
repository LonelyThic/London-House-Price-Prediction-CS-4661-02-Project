import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.baseline import (CAT_COLS, NUM_COLS, PREV_SALE_COLS, encode,
                          median_baseline)
from src.data import load_test, load_train, time_split
from src.features import add_previous_sale, build_price_index, month_index

OUT = Path(__file__).resolve().parent / "reports"
FIG = OUT / "figures"
SEEDS = [42, 1, 7]

# Chart styling: one series per chart, so one hue; recessive axes and grid.
BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def gb_predict(train, valid, num_cols, seed):
    """Same model as src.baseline.gradient_boosting, with a chosen seed."""
    X_train, X_valid = encode(train, valid, num_cols)
    model = HistGradientBoostingRegressor(
        max_iter=800, learning_rate=0.08, max_leaf_nodes=127,
        categorical_features=list(range(len(CAT_COLS))), random_state=seed,
    )
    model.fit(X_train, np.log(train["price"]))
    return np.exp(model.predict(X_valid))


def coverage(frame):
    return float(frame["prev_price"].notna().mean())


def leak_violations(frame):
    """Rows whose 'previous' sale is not from a strictly earlier month."""
    has_prev = frame["prev_price"].notna()
    return int((frame.loc[has_prev, "months_since_prev_sale"] < 1).sum())


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    full = load_train()
    train, valid = time_split(full, valid_from_year=2023)
    train_f, valid_f = add_previous_sale(train, valid)
    _, test_f = add_previous_sale(full, load_test())

    # --- Data and feature summary -------------------------------------
    summary = {
        "training_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test_f),
        "training_years": [int(train.sale_year.min()), int(train.sale_year.max())],
        "validation_year": 2023,
        "share_with_previous_sale": {
            "train_before_2023": round(coverage(train_f), 4),
            "validation_2023": round(coverage(valid_f), 4),
            "test_2024": round(coverage(test_f), 4),
        },
        "leak_check_violations": {
            "train_before_2023": leak_violations(train_f),
            "validation_2023": leak_violations(valid_f),
            "test_2024": leak_violations(test_f),
        },
        "median_months_since_previous_sale_validation":
            float(valid_f["months_since_prev_sale"].median()),
    }
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=2))

    # --- London price index (training data only) -----------------------
    index = build_price_index(train)
    months = index.index.to_numpy()
    index_table = pd.DataFrame({
        "sale_year": months // 12,
        "sale_month": months % 12 + 1,
        "log_price_index": index.to_numpy(),
        "typical_price_gbp": np.exp(index.to_numpy()).round(0),
    })
    index_table.to_csv(OUT / "price_index.csv", index=False)

    # --- Model comparison (seed 42, same as src.baseline) --------------
    rows = []
    jobs = [
        ("Median by outcode + type", lambda: median_baseline(train, valid)),
        ("Gradient boosting", lambda: gb_predict(train, valid, NUM_COLS, 42)),
        ("Gradient boosting + previous sale",
         lambda: gb_predict(train_f, valid_f, NUM_COLS + PREV_SALE_COLS, 42)),
    ]
    for name, run in jobs:
        mae = mean_absolute_error(valid["price"], run())
        rows.append({"model": name, "mae_gbp": round(mae, 2),
                     "training_rows": len(train), "validation_rows": len(valid)})
        print(f"{name:<36} MAE: £{mae:,.0f}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "model_comparison.csv", index=False)

    # --- Seed check: how much does MAE move by chance? -----------------
    seed_rows = []
    for seed in SEEDS:
        for name, (a, b, cols) in {
            "Gradient boosting": (train, valid, NUM_COLS),
            "Gradient boosting + previous sale":
                (train_f, valid_f, NUM_COLS + PREV_SALE_COLS),
        }.items():
            mae = mean_absolute_error(valid["price"], gb_predict(a, b, cols, seed))
            seed_rows.append({"model": name, "seed": seed, "mae_gbp": round(mae, 2)})
            print(f"seed {seed:<3} {name:<36} MAE: £{mae:,.0f}", flush=True)
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(OUT / "seed_check.csv", index=False)
    seeds.groupby("model")["mae_gbp"].agg(["mean", "min", "max"]).round(2).to_csv(
        OUT / "seed_check_summary.csv")

    # --- Figures --------------------------------------------------------
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    order = results.sort_values("mae_gbp", ascending=True)
    ax.barh(order["model"], order["mae_gbp"] / 1000, color=BLUE, height=0.5)
    for y, v in enumerate(order["mae_gbp"] / 1000):
        ax.text(v + 4, y, f"£{v:,.0f}k", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Validation MAE on 2023 sales (£ thousands, lower is better)")
    ax.set_title("Model comparison: same training and validation rows", loc="left")
    ax.set_xlim(0, order["mae_gbp"].max() / 1000 * 1.15)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    dates = pd.to_datetime(dict(year=index_table.sale_year,
                                month=index_table.sale_month, day=1))
    ax.plot(dates, index_table["typical_price_gbp"] / 1000, color=BLUE, linewidth=2)
    last = index_table.iloc[-1]
    ax.annotate(f"£{last.typical_price_gbp / 1000:,.0f}k",
                (dates.iloc[-1], last.typical_price_gbp / 1000),
                xytext=(6, 0), textcoords="offset points", va="center",
                color=INK, fontsize=9)
    ax.set_ylabel("Typical sale price (£ thousands)")
    ax.set_title("London price index from training sales (12-month rolling median, "
                 "earlier months only)", loc="left")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "price_index.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.0), facecolor=SURFACE)
    cov = pd.Series(summary["share_with_previous_sale"]) * 100
    labels = ["Training (before 2023)", "Validation (2023)", "Test (2024)"]
    ax.barh(labels, cov.to_numpy(), color=BLUE, height=0.5)
    for y, v in enumerate(cov.to_numpy()):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of sales whose property sold in an earlier month (%)")
    ax.set_title("How often a previous sale is available", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "previous_sale_coverage.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()