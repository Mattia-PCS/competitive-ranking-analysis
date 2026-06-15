from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import FIGURES_DIR, TABLES_DIR


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def save_table(df: pd.DataFrame, name: str) -> None:
    ensure_output_dirs()
    df.to_csv(TABLES_DIR / f"{name}.csv", index=True)
    try:
        df.to_latex(TABLES_DIR / f"{name}.tex", index=True, float_format="%.4f")
    except Exception:
        pass


def save_figure(name: str) -> None:
    ensure_output_dirs()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def standardize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = df[columns].copy().astype(float)
    return (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
