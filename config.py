"""Central configuration for the xG World Cup project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WC2026_DIR = DATA_DIR / "wc2026"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

SHOTS_PARQUET = DATA_DIR / "shots.parquet"

# StatsBomb open-data tournaments: (competition_id, season_id, label)
# To add the 2026 World Cup once StatsBomb releases it, append its
# (43, <season_id>) entry here and re-run scripts/01_build_dataset.py.
TOURNAMENTS = [
    (43, 3, "Men's WC 2018"),
    (43, 106, "Men's WC 2022"),
    (72, 30, "Women's WC 2019"),
    (72, 107, "Women's WC 2023"),
]

# FIFA World Cup 2026 aggregate dataset (CC0, github.com/mominullptr)
WC2026_REPO_RAW = (
    "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main"
)
WC2026_FILES = [
    "matches.csv",
    "matches_detailed.csv",
    "match_team_stats.csv",
    "match_events.csv",
    "teams.csv",
    "tournament_stages.csv",
]

# StatsBomb pitch coordinates: x in [0, 120], y in [0, 80], attacking goal at x=120
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_CENTER = (120.0, 40.0)
POST_LEFT = (120.0, 36.0)   # smaller y post
POST_RIGHT = (120.0, 44.0)  # larger y post

# Residual pitch grid (attacking area x >= 60), 5x5-yard cells
GRID_X_MIN = 60.0
GRID_CELL = 5.0
MIN_SHOTS_PER_BIN = 20  # bins below this are merged/dropped from tests

SEED = 42
N_FOLDS = 5           # GroupKFold by match_id
N_BOOTSTRAP = 2000    # CI resamples for metrics
N_PERMUTATIONS = 9999  # Moran's I permutation test

# Subgroup definitions
TIGHT_ANGLE_RAD = 0.26  # ~15 degrees: "byline / tight angle" zone
