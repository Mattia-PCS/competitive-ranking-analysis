from __future__ import annotations

import pandas as pd
from .config import COMMON_FEATURES
from .utils import available_columns, save_table


def run(players_by_dataset: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, df in players_by_dataset.items():
        features = available_columns(df, COMMON_FEATURES)
        for f in features:
            rows.append({
                "dataset": name,
                "feature": f,
                "mean": df[f].mean(),
                "median": df[f].median(),
                "std": df[f].std(),
                "min": df[f].min(),
                "max": df[f].max(),
            })
    save_table(pd.DataFrame(rows), "community_comparison")
