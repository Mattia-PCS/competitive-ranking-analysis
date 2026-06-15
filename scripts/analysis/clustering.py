from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .config import COMMON_FEATURES, BLOCKBALL_EXTRA_FEATURES
from .utils import available_columns, standardize, save_table, save_figure


def run(players_by_dataset: dict[str, pd.DataFrame], k: int = 4) -> None:
    metric_rows = []
    profile_tables = []
    for name, df in players_by_dataset.items():
        features = available_columns(df, COMMON_FEATURES + BLOCKBALL_EXTRA_FEATURES)
        features = [f for f in features if f != "current_mmr"]
        if len(features) < 2:
            continue
        x = standardize(df, features).dropna()
        if len(x) <= k:
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x)
        sil = silhouette_score(x, labels) if k > 1 else None
        metric_rows.append({"dataset": name, "k": k, "silhouette_score": sil})

        clustered = df.loc[x.index].copy()
        clustered["cluster"] = labels
        profiles = clustered.groupby("cluster")[available_columns(clustered, features + ["current_mmr"])].mean()
        profiles.insert(0, "dataset", name)
        profile_tables.append(profiles)

        # Small internal PCA only for visualization of clusters.
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(x)
        plt.figure(figsize=(7, 5))
        plt.scatter(coords[:, 0], coords[:, 1], c=labels, alpha=0.8)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{name}: K-Means clusters in PCA space")
        save_figure(f"{name.lower()}_clusters_pca")

    if metric_rows:
        save_table(pd.DataFrame(metric_rows), "clustering_metrics")
    if profile_tables:
        save_table(pd.concat(profile_tables), "cluster_profiles")
