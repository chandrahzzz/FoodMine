"""Paths and project-wide constants for FoodieMine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "zomato.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_CSV = PROCESSED_DIR / "restaurants_clean.csv"
UNRATED_CSV = PROCESSED_DIR / "restaurants_unrated.csv"
WAREHOUSE_DB = DATA_DIR / "warehouse" / "foodiemine.db"

OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

RANDOM_STATE = 42

# A restaurant is "Highly Rated" when its rating is at least this value.
HIGH_RATING_THRESHOLD = 4.0

# Discretisation boundaries (cost for two people, in rupees).
COST_BINS = [0, 400, 800, float("inf")]
COST_LABELS = ["Low", "Medium", "High"]

# Rating levels: Poor < 3.5 <= Average < 4.0 <= Good.
RATING_BINS = [0, 3.5, 4.0, 5.01]
RATING_LABELS = ["Poor", "Average", "Good"]

# Chart palette (validated categorical slots + sequential blue ramp).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRAY = "#8a8985"


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, WAREHOUSE_DB.parent, FIG_DIR, TABLE_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
