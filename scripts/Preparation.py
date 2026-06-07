import pandas as pd
import numpy as np

MMR_HISTORY = "MMR_HISTORY.csv"
PLAYERS_AND_MMR = "PLAYERS_AND_MMR.csv"
RANKED_STATS = "RANKED_STATS.csv"

PLAYERS_OUTPUT = "players_clean.csv"
MMR_HISTORY_OUTPUT = "mmr_history_clean.csv"
PLAYER_MAP = "private_uuid_player_map.csv"

# -----------------------
# Load
# -----------------------

mmr_history = pd.read_csv(MMR_HISTORY)
players = pd.read_csv(PLAYERS_AND_MMR)
stats = pd.read_csv(RANKED_STATS)

# -----------------------
# Anonymize UUIDs
# -----------------------

all_uuids = sorted(
    set(players["uuid"]) |
    set(stats["uuid"]) |
    set(mmr_history["uuid"])
)

uuid_map = {
    uuid: f"Player_{i+1:03d}"
    for i, uuid in enumerate(all_uuids)
}

for df in [mmr_history, players, stats]:
    df["player"] = df["uuid"].map(uuid_map)
    df.drop(columns=["uuid"], inplace=True)

pd.DataFrame({
    "uuid": list(uuid_map.keys()),
    "player": list(uuid_map.values())
}).to_csv(PLAYER_MAP, index=False)

# -----------------------
# Clean PLAYERS_AND_MMR
# -----------------------

players = players.drop(columns=["banned", "ranked_ban_until"], errors="ignore")
players = players.rename(columns={"mmr": "current_mmr"})

# -----------------------
# Pivot RANKED_STATS
# -----------------------

stats_wide = stats.pivot_table(
    index="player",
    columns="stat",
    values="value",
    aggfunc="sum",
    fill_value=0
).reset_index()

stats_wide.columns.name = None
stats_wide.columns = [str(c).lower() for c in stats_wide.columns]

expected_cols = [
    "goals",
    "assists",
    "saves",
    "wins",
    "losses",
    "draws",
    "matches_played"
]

for col in expected_cols:
    if col not in stats_wide.columns:
        stats_wide[col] = 0

# -----------------------
# Create player-level dataset
# -----------------------

players_clean = players[["player", "current_mmr"]].merge(
    stats_wide,
    on="player",
    how="left"
)

players_clean[expected_cols] = players_clean[expected_cols].fillna(0)

# Remove players with 0 matches
players_clean = players_clean[players_clean["matches_played"] > 0].copy()

# -----------------------
# Feature engineering
# -----------------------

players_clean["win_rate"] = (
    players_clean["wins"] / players_clean["matches_played"]
)

players_clean["goals_per_match"] = (
    players_clean["goals"] / players_clean["matches_played"]
)

players_clean["assists_per_match"] = (
    players_clean["assists"] / players_clean["matches_played"]
)

players_clean["saves_per_match"] = (
    players_clean["saves"] / players_clean["matches_played"]
)

players_clean = players_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

# -----------------------
# Reorder player columns
# -----------------------

player_cols = [
    "player",
    "current_mmr",
    "matches_played",
    "wins",
    "losses",
    "draws",
    "goals",
    "assists",
    "saves",
    "win_rate",
    "goals_per_match",
    "assists_per_match",
    "saves_per_match"
]

players_clean = players_clean[player_cols]

# -----------------------
# Clean MMR history dataset
# -----------------------

mmr_history = mmr_history.drop(columns=["id"], errors="ignore")

mmr_history["created_at"] = pd.to_datetime(
    mmr_history["created_at"],
    errors="coerce"
)

mmr_history = mmr_history.dropna(subset=["created_at"])
mmr_history = mmr_history.sort_values(["player", "created_at"])

mmr_history["first_recorded_at"] = (
    mmr_history.groupby("player")["created_at"].transform("min")
)

mmr_history["days_from_first_record"] = (
    mmr_history["created_at"] - mmr_history["first_recorded_at"]
).dt.total_seconds() / 86400

mmr_history_clean = mmr_history[
    [
        "player",
        "created_at",
        "days_from_first_record",
        "mmr"
    ]
].copy()

# Keep only players that are also in players_clean
valid_players = set(players_clean["player"])
mmr_history_clean = mmr_history_clean[
    mmr_history_clean["player"].isin(valid_players)
].copy()

# -----------------------
# Save outputs
# -----------------------

players_clean.to_csv(PLAYERS_OUTPUT, index=False)
mmr_history_clean.to_csv(MMR_HISTORY_OUTPUT, index=False)

print(f"Saved: {PLAYERS_OUTPUT}")
print(f"Rows: {len(players_clean)}")
print(f"Players: {players_clean['player'].nunique()}")

print()

print(f"Saved: {MMR_HISTORY_OUTPUT}")
print(f"Rows: {len(mmr_history_clean)}")
print(f"Players: {mmr_history_clean['player'].nunique()}")

print()

print(f"Private map saved: {PLAYER_MAP}")