from pathlib import Path

# Change BASE_DIR to your local folder containing the two dataset folders.
# Example on Windows:
# BASE_DIR = Path(r"C:\Users\matti\Desktop\Università\Secondo anno Sapienza\Data Unit 2\Data tables")
BASE_DIR = Path("data")

FOOTBALL_DIR = BASE_DIR / "Football"
BLOCKBALL_DIR = BASE_DIR / "BlockBall"

OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

DATASETS = {
    "Community A": {
        "players": FOOTBALL_DIR / "players_clean.csv",
        "history": FOOTBALL_DIR / "mmr_history_clean.csv",
    },
    "Community B": {
        "players": BLOCKBALL_DIR / "bb_players_clean.csv",
        "history": BLOCKBALL_DIR / "bb_mmr_history_clean.csv",
    },
}

COMMON_FEATURES = [
    "current_mmr",
    "matches_played",
    "win_rate",
    "goals_per_match",
    "assists_per_match",
    "saves_per_match",
]

BLOCKBALL_EXTRA_FEATURES = ["passes_per_match"]
