from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from .config import COMMON_FEATURES, BLOCKBALL_EXTRA_FEATURES
from .utils import available_columns, save_table, save_figure


def run(players_by_dataset: dict[str, pd.DataFrame]) -> None:
    all_group_stats = []
    for name, df in players_by_dataset.items():
        if "current_mmr" not in df.columns:
            continue
        data = df.copy()
        data["rank_group"] = pd.qcut(
            data["current_mmr"],
            q=[0, 0.25, 0.75, 1.0],
            labels=["Bottom 25%", "Middle 50%", "Top 25%"],
            duplicates="drop",
        )
        features = available_columns(data, COMMON_FEATURES + BLOCKBALL_EXTRA_FEATURES)
        stats = data.groupby("rank_group", observed=True)[features].mean()
        stats.insert(0, "dataset", name)
        all_group_stats.append(stats)

        plot_features = [f for f in ["win_rate", "goals_per_match", "assists_per_match", "saves_per_match"] if f in data.columns]
        for feature in plot_features:
            groups = [g[feature].dropna().values for _, g in data.groupby("rank_group", observed=True)]
            labels = [str(k) for k, _ in data.groupby("rank_group", observed=True)]
            plt.figure(figsize=(7, 4))
            plt.boxplot(groups, labels=labels)
            plt.ylabel(feature)
            plt.title(f"{name}: {feature} by rank group")
            save_figure(f"{name.lower()}_{feature}_by_rank_group")

    if all_group_stats:
        save_table(pd.concat(all_group_stats), "rank_group_means")
