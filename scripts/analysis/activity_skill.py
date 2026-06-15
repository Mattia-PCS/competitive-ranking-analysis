from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from .utils import save_table, save_figure


def run(players_by_dataset: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, df in players_by_dataset.items():
        if {"matches_played", "current_mmr"}.issubset(df.columns):
            corr = df[["matches_played", "current_mmr"]].corr().iloc[0, 1]
            rows.append({"dataset": name, "relationship": "matches_played_vs_current_mmr", "pearson_corr": corr})
            plt.figure(figsize=(7, 4))
            plt.scatter(df["matches_played"], df["current_mmr"], alpha=0.75)
            plt.xlabel("Matches played")
            plt.ylabel("Current MMR")
            plt.title(f"{name}: activity vs rating")
            save_figure(f"{name.lower()}_activity_vs_mmr")

        if {"matches_played", "win_rate"}.issubset(df.columns):
            corr = df[["matches_played", "win_rate"]].corr().iloc[0, 1]
            rows.append({"dataset": name, "relationship": "matches_played_vs_win_rate", "pearson_corr": corr})

    if rows:
        save_table(pd.DataFrame(rows), "activity_skill_correlations")
