from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from .utils import numeric_columns, save_table, save_figure


def _heatmap(corr: pd.DataFrame, title: str, name: str) -> None:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    save_figure(name)


def run(players_by_dataset: dict[str, pd.DataFrame]) -> None:
    top_rows = []
    for name, df in players_by_dataset.items():
        cols = numeric_columns(df)
        corr = df[cols].corr(method="pearson")
        save_table(corr, f"{name.lower()}_correlation_matrix")
        _heatmap(corr, f"{name}: Pearson correlation matrix", f"{name.lower()}_correlation_heatmap")

        if "current_mmr" in corr.columns:
            mmr_corr = corr["current_mmr"].drop("current_mmr", errors="ignore").sort_values(key=abs, ascending=False)
            for feature, value in mmr_corr.items():
                top_rows.append({"dataset": name, "feature": feature, "pearson_corr_with_mmr": value})

    if top_rows:
        save_table(pd.DataFrame(top_rows), "top_correlations_with_mmr")
