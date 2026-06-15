from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .config import DATASETS
from .utils import read_csv, numeric_columns, save_figure


THRESHOLDS = [0, 20, 30, 40, 50]


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def plot_heatmap(corr: pd.DataFrame, title: str, output_name: str) -> None:
    plt.figure(figsize=(9, 7))

    im = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=45,
        ha="right",
        fontsize=8
    )

    plt.yticks(
        range(len(corr.index)),
        corr.index,
        fontsize=8
    )

    plt.title(title, fontsize=11)

    save_figure(output_name)


def main() -> None:
    for dataset_name, paths in DATASETS.items():
        df = read_csv(paths["players"])

        if "matches_played" not in df.columns:
            print(f"[SKIP] {dataset_name}: missing matches_played column")
            continue

        for threshold in THRESHOLDS:
            if threshold == 0:
                filtered = df.copy()
                label = "all players"
                suffix = "all"
            else:
                filtered = df[df["matches_played"] >= threshold].copy()
                label = f"{threshold}+ matches"
                suffix = f"{threshold}plus"

            print(
                f"{dataset_name} | {label}: "
                f"{len(df)} players -> {len(filtered)} players"
            )

            if len(filtered) < 3:
                print(f"[SKIP] {dataset_name} | {label}: not enough players")
                continue

            cols = numeric_columns(filtered)

            if len(cols) < 2:
                print(f"[SKIP] {dataset_name} | {label}: not enough numeric columns")
                continue

            corr = filtered[cols].corr(method="pearson")

            output_name = (
                f"{safe_name(dataset_name)}_correlation_heatmap_{suffix}"
            )

            plot_heatmap(
                corr,
                f"{dataset_name}: Pearson correlation matrix ({label})",
                output_name
            )


if __name__ == "__main__":
    main()