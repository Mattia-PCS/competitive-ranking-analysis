from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from .utils import save_table, save_figure


def run(history_by_dataset: dict[str, pd.DataFrame]) -> None:
    stability_rows = []
    for name, df in history_by_dataset.items():
        required = {"player", "days_from_first_record", "mmr"}
        if not required.issubset(df.columns):
            continue
        data = df.sort_values(["player", "days_from_first_record"])

        # Representative trajectories: players with the highest number of observations.
        top_players = data["player"].value_counts().head(8).index
        plt.figure(figsize=(8, 5))
        for p in top_players:
            s = data[data["player"] == p]
            plt.plot(s["days_from_first_record"], s["mmr"], alpha=0.8, label=str(p))
        plt.xlabel("Days from first record")
        plt.ylabel("MMR")
        plt.title(f"{name}: representative MMR trajectories")
        plt.legend(fontsize=7)
        save_figure(f"{name.lower()}_mmr_trajectories")

        # Aggregate trend by rounded day.
        data["day_bin"] = data["days_from_first_record"].round().astype(int)
        trend = data.groupby("day_bin")["mmr"].mean().reset_index()
        trend.to_csv(f"output/tables/{name.lower()}_aggregate_mmr_trend.csv", index=False)
        plt.figure(figsize=(8, 4))
        plt.plot(trend["day_bin"], trend["mmr"])
        plt.xlabel("Days from first record")
        plt.ylabel("Average MMR")
        plt.title(f"{name}: aggregate MMR trend")
        save_figure(f"{name.lower()}_aggregate_mmr_trend")

        grouped = data.groupby("player")["mmr"]
        stats = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
        stats["mmr_swing"] = stats["max"] - stats["min"]
        stats.insert(0, "dataset", name)
        stability_rows.append(stats)

        plt.figure(figsize=(7, 4))
        plt.hist(stats["std"].dropna(), bins=20)
        plt.xlabel("MMR standard deviation")
        plt.ylabel("Players")
        plt.title(f"{name}: rating volatility")
        save_figure(f"{name.lower()}_rating_volatility")

    if stability_rows:
        save_table(pd.concat(stability_rows), "rating_stability")
