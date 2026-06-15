from __future__ import annotations

import argparse
from pathlib import Path
import sklearn

from analysis.config import DATASETS, OUTPUT_DIR
from analysis.utils import ensure_output_dirs, read_csv
from analysis import descriptive, correlations, rank_groups, activity_skill
from analysis import community_comparison, pca_analysis, clustering, temporal, outliers


def load_datasets() -> tuple[dict, dict]:
    players_by_dataset = {}
    history_by_dataset = {}
    for name, paths in DATASETS.items():
        players_by_dataset[name] = read_csv(paths["players"])
        history_by_dataset[name] = read_csv(paths["history"])
    return players_by_dataset, history_by_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ranking-system data analysis.")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters for K-Means.")
    args = parser.parse_args()

    ensure_output_dirs()
    players_by_dataset, history_by_dataset = load_datasets()

    descriptive.run(players_by_dataset)
    correlations.run(players_by_dataset)
    rank_groups.run(players_by_dataset)
    activity_skill.run(players_by_dataset)
    community_comparison.run(players_by_dataset)
    pca_analysis.run(players_by_dataset)
    clustering.run(players_by_dataset, k=args.k)
    temporal.run(history_by_dataset)
    outliers.run(players_by_dataset)

    print(f"Analysis completed. Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
