import pandas as pd
import numpy as np

BB_ELO_HISTORY = "BB_elohistory.csv"
BB_PLAYER_STATS = "BB_playersta.csv"
BB_UUID_TO_NAME = "BB_UuidToName.csv"

PLAYERS_OUTPUT = "bb_players_clean.csv"
MMR_HISTORY_OUTPUT = "bb_mmr_history_clean.csv"
PLAYER_MAP = "bb_private_uuid_player_map.csv"

MAX_VALID_MMR = 5000

# -----------------------
# Load
# -----------------------

elo_history = pd.read_csv(BB_ELO_HISTORY)
players = pd.read_csv(BB_PLAYER_STATS)
uuid_names = pd.read_csv(BB_UUID_TO_NAME)

# -----------------------
# Standardize column names
# -----------------------

elo_history.columns = [c.lower().strip() for c in elo_history.columns]
players.columns = [c.lower().strip() for c in players.columns]
uuid_names.columns = [c.lower().strip() for c in uuid_names.columns]

# -----------------------
# Detect UUID columns
# -----------------------

def find_uuid_column(df):
    for possible in ["uuid", "player_uuid", "playeruuid", "player_id"]:
        if possible in df.columns:
            return possible
    raise ValueError(f"No UUID column found. Columns: {list(df.columns)}")


elo_uuid_col = find_uuid_column(elo_history)
players_uuid_col = find_uuid_column(players)
names_uuid_col = find_uuid_column(uuid_names)

# -----------------------
# Anonymize UUIDs globally
# -----------------------

all_uuids = sorted(
    set(elo_history[elo_uuid_col].dropna().astype(str)) |
    set(players[players_uuid_col].dropna().astype(str)) |
    set(uuid_names[names_uuid_col].dropna().astype(str))
)

uuid_map = {
    uuid: f"BB_Player_{i + 1:03d}"
    for i, uuid in enumerate(all_uuids)
}

elo_history["player"] = elo_history[elo_uuid_col].astype(str).map(uuid_map)
players["player"] = players[players_uuid_col].astype(str).map(uuid_map)
uuid_names["player"] = uuid_names[names_uuid_col].astype(str).map(uuid_map)

pd.DataFrame({
    "uuid": list(uuid_map.keys()),
    "player": list(uuid_map.values())
}).to_csv(PLAYER_MAP, index=False)

elo_history = elo_history.drop(columns=[elo_uuid_col], errors="ignore")
players = players.drop(columns=[players_uuid_col], errors="ignore")
uuid_names = uuid_names.drop(columns=[names_uuid_col], errors="ignore")

# -----------------------
# Clean ELO/MMR history
# -----------------------

elo_history = elo_history.drop(columns=["id"], errors="ignore")

if "elo" in elo_history.columns:
    elo_history = elo_history.rename(columns={"elo": "mmr"})

timestamp_col = None
for possible in ["created_at", "createdat", "timestamp", "date", "time"]:
    if possible in elo_history.columns:
        timestamp_col = possible
        break

if timestamp_col is None:
    raise ValueError(f"No timestamp column found. Columns: {list(elo_history.columns)}")

elo_history["created_at"] = pd.to_datetime(
    elo_history[timestamp_col],
    errors="coerce"
)

if timestamp_col != "created_at":
    elo_history = elo_history.drop(columns=[timestamp_col], errors="ignore")

if "mmr" not in elo_history.columns:
    raise ValueError(f"No MMR/ELO column found. Columns: {list(elo_history.columns)}")

elo_history["mmr"] = pd.to_numeric(elo_history["mmr"], errors="coerce")

elo_history = elo_history.dropna(subset=["player", "created_at", "mmr"])

# Remove impossible MMR/ELO values
invalid_history = elo_history[elo_history["mmr"] >= MAX_VALID_MMR].copy()
elo_history = elo_history[elo_history["mmr"] < MAX_VALID_MMR].copy()

print(f"Removed impossible MMR history records: {len(invalid_history)}")

elo_history = elo_history.sort_values(["player", "created_at"])

elo_history["first_recorded_at"] = (
    elo_history.groupby("player")["created_at"].transform("min")
)

elo_history["days_from_first_record"] = (
    elo_history["created_at"] - elo_history["first_recorded_at"]
).dt.total_seconds() / 86400

mmr_history_clean = elo_history[
    [
        "player",
        "created_at",
        "days_from_first_record",
        "mmr"
    ]
].copy()

# -----------------------
# Clean player-level dataset
# -----------------------

if "mmr" in players.columns:
    players = players.rename(columns={"mmr": "current_mmr"})
elif "elo" in players.columns:
    players = players.rename(columns={"elo": "current_mmr"})

if "games" in players.columns:
    players = players.rename(columns={"games": "matches_played"})

players = players.drop(
    columns=[
        "name",
        "ign",
        "username",
        "language",
        "coins",
        "xp",
        "level",
        "last_reward_claimed_level",
        "ranked_ban_until",
        "banned",
        "winrate"
    ],
    errors="ignore"
)

expected_cols = [
    "current_mmr",
    "matches_played",
    "wins",
    "losses",
    "draws",
    "goals",
    "assists",
    "passes",
    "saves",
    "peak_mmr"
]

for col in expected_cols:
    if col not in players.columns:
        players[col] = 0

for col in expected_cols:
    players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0)

players_clean = players[players["matches_played"] > 0].copy()

# -----------------------
# Recompute peak_mmr from cleaned history
# -----------------------

peak_from_history = (
    mmr_history_clean
    .groupby("player")["mmr"]
    .max()
)

players_clean["peak_mmr"] = players_clean["player"].map(peak_from_history)

players_clean["peak_mmr"] = players_clean["peak_mmr"].fillna(
    players_clean["current_mmr"]
)

# Optional: also clean impossible current_mmr if ever present
players_clean.loc[
    players_clean["current_mmr"] >= MAX_VALID_MMR,
    "current_mmr"
] = players_clean.loc[
    players_clean["current_mmr"] >= MAX_VALID_MMR,
    "player"
].map(peak_from_history)

players_clean["current_mmr"] = players_clean["current_mmr"].fillna(
    players_clean["peak_mmr"]
)

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

players_clean["passes_per_match"] = (
    players_clean["passes"] / players_clean["matches_played"]
)

players_clean["saves_per_match"] = (
    players_clean["saves"] / players_clean["matches_played"]
)

players_clean = players_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

# -----------------------
# Keep only valid players in history
# -----------------------

valid_players = set(players_clean["player"])

mmr_history_clean = mmr_history_clean[
    mmr_history_clean["player"].isin(valid_players)
].copy()

# -----------------------
# Reorder player columns
# -----------------------

preferred_cols = [
    "player",
    "current_mmr",
    "peak_mmr",
    "matches_played",
    "wins",
    "losses",
    "draws",
    "goals",
    "assists",
    "passes",
    "saves",
    "win_rate",
    "goals_per_match",
    "assists_per_match",
    "passes_per_match",
    "saves_per_match"
]

extra_cols = [
    col for col in players_clean.columns
    if col not in preferred_cols
]

players_clean = players_clean[preferred_cols + extra_cols]

# -----------------------
# Save outputs
# -----------------------

players_clean.to_csv(PLAYERS_OUTPUT, index=False)
mmr_history_clean.to_csv(MMR_HISTORY_OUTPUT, index=False)

print()
print(f"Saved: {PLAYERS_OUTPUT}")
print(f"Rows: {len(players_clean)}")
print(f"Players: {players_clean['player'].nunique()}")
print(f"Max current_mmr: {players_clean['current_mmr'].max()}")
print(f"Max peak_mmr: {players_clean['peak_mmr'].max()}")

print()

print(f"Saved: {MMR_HISTORY_OUTPUT}")
print(f"Rows: {len(mmr_history_clean)}")
print(f"Players: {mmr_history_clean['player'].nunique()}")
print(f"Max history mmr: {mmr_history_clean['mmr'].max()}")

print()

print(f"Private map saved: {PLAYER_MAP}")

# -----------------------
# Consistency check
# -----------------------

history_players = set(mmr_history_clean["player"])
all_players = set(players_clean["player"])

missing_history = all_players - history_players

print()
print(f"Players without MMR/ELO history: {len(missing_history)}")

if missing_history:
    print("First missing players:")
    for p in sorted(list(missing_history))[:20]:
        row = players_clean[players_clean["player"] == p].iloc[0]
        print(
            p,
            "| matches =", int(row["matches_played"]),
            "| current_mmr =", float(row["current_mmr"])
        )