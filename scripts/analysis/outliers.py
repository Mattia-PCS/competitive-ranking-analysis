from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from .utils import save_table, save_figure


def run(players_by_dataset: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, df in players_by_dataset.items():
        if not {"current_mmr", "win_rate"}.issubset(df.columns):
            continue
        data = df.copy()
        data["mmr_z"] = (data["current_mmr"] - data["current_mmr"].mean()) / data["current_mmr"].std(ddof=0)
        data["win_rate_z"] = (data["win_rate"] - data["win_rate"].mean()) / data["win_rate"].std(ddof=0)
        data["outlier_score"] = (data["mmr_z"] - data["win_rate_z"]).abs()
        top = data.sort_values("outlier_score", ascending=False).head(10)
        top.insert(0, "dataset", name)
        rows.append(top[[c for c in ["dataset", "player", "current_mmr", "win_rate", "matches_played", "outlier_score"] if c in top.columns]])

        plt.figure(figsize=(7, 4))
        plt.scatter(data["win_rate"], data["current_mmr"], alpha=0.75)
        plt.xlabel("Win rate")
        plt.ylabel("Current MMR")
        plt.title(f"{name}: MMR vs win rate outliers")
        save_figure(f"{name.lower()}_outliers_mmr_winrate")

    if rows:
        save_table(pd.concat(rows), "outlier_candidates")
