from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .config import COMMON_FEATURES, BLOCKBALL_EXTRA_FEATURES
from .utils import available_columns, standardize, save_table, save_figure


def run(players_by_dataset: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    projections = {}
    variance_rows = []
    for name, df in players_by_dataset.items():
        features = available_columns(df, COMMON_FEATURES + BLOCKBALL_EXTRA_FEATURES)
        features = [f for f in features if f != "current_mmr"]
        if len(features) < 2:
            continue
        x = standardize(df, features).dropna()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(x)
        proj = pd.DataFrame(coords, columns=["PC1", "PC2"], index=x.index)
        proj["player"] = df.loc[x.index, "player"].values if "player" in df.columns else x.index.astype(str)
        proj["current_mmr"] = df.loc[x.index, "current_mmr"].values if "current_mmr" in df.columns else None
        projections[name] = proj

        for i, v in enumerate(pca.explained_variance_ratio_, start=1):
            variance_rows.append({"dataset": name, "component": f"PC{i}", "explained_variance_ratio": v})

        plt.figure(figsize=(7, 5))
        plt.scatter(proj["PC1"], proj["PC2"], alpha=0.75)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{name}: PCA projection")
        save_figure(f"{name.lower()}_pca_projection")

        loadings = pd.DataFrame(pca.components_.T, index=features, columns=["PC1", "PC2"])
        save_table(loadings, f"{name.lower()}_pca_loadings")

    if variance_rows:
        save_table(pd.DataFrame(variance_rows), "pca_explained_variance")
    return projections
