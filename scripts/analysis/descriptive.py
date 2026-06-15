from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from .utils import numeric_columns, save_table, save_figure


def run(players_by_dataset: dict[str, pd.DataFrame]) -> None:
    summaries = []
    for name, df in players_by_dataset.items():
        nums = numeric_columns(df)
        summary = df[nums].describe().T
        summary.insert(0, "dataset", name)
        summaries.append(summary)

        if "current_mmr" in df.columns:
            plt.figure(figsize=(7, 4))
            plt.hist(df["current_mmr"].dropna(), bins=20)
            plt.xlabel("Current MMR")
            plt.ylabel("Players")
            plt.title(f"{name}: MMR distribution")
            save_figure(f"{name.lower()}_mmr_distribution")

        if "matches_played" in df.columns:
            plt.figure(figsize=(7, 4))
            plt.hist(df["matches_played"].dropna(), bins=20)
            plt.xlabel("Matches played")
            plt.ylabel("Players")
            plt.title(f"{name}: activity distribution")
            save_figure(f"{name.lower()}_matches_distribution")

    if summaries:
        save_table(pd.concat(summaries), "summary_statistics")
