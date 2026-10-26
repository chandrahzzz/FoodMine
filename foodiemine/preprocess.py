"""Step 1 - Data preprocessing (Module 3).

Raw Zomato listings -> one clean row per restaurant, ready for mining.

* drop columns not needed (url, phone, reviews, menu, dishes)
* clean `rate` ("4.1/5", "3.9 /5", "NEW", "-") and `approx_cost` ("1,200")
* remove duplicates: the same restaurant appears once per listing category
  (Delivery, Dine-out, ...) and per city section, so rows are merged on
  (name, address) and the listing categories are kept as a set
* handle missing values
* derive hierarchy attributes (area, zone, cuisine groups)
* discretise cost / rating / votes and create the classification label
"""
import json

import numpy as np
import pandas as pd

from . import config
from .mappings import LOCATION_TO_ZONE, cuisine_to_group, location_to_area

USE_COLS = [
    "name", "address", "online_order", "book_table", "rate", "votes", "location",
    "rest_type", "cuisines", "approx_cost(for two people)", "listed_in(type)",
]
LIST_COLS = ["cuisine_list", "cuisine_groups", "rest_types", "listed_types"]


def clean_rate(s: pd.Series) -> pd.Series:
    """'4.1/5' -> 4.1, '3.9 /5' -> 3.9, 'NEW' / '-' / NaN -> NaN."""
    s = s.astype("string").str.replace("/5", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def clean_cost(s: pd.Series) -> pd.Series:
    """'1,200' -> 1200.0"""
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False), errors="coerce")


def split_list(s: pd.Series) -> pd.Series:
    return s.fillna("").map(lambda v: [x.strip() for x in str(v).split(",") if x.strip()])


# A restaurant listed under several categories gets the most specific one
# as its service type (a pub that also delivers is still a pub).
SERVICE_PRIORITY = [
    "Pubs and bars", "Drinks & nightlife", "Buffet", "Cafes", "Desserts",
    "Dine-out", "Delivery",
]


def _primary_service(values: pd.Series) -> str:
    present = set(values)
    return next(s for s in SERVICE_PRIORITY if s in present)


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, usecols=USE_COLS)


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (rated restaurants, unrated/new restaurants, cleaning report)."""
    report = {"raw_rows": len(raw), "raw_columns_kept": USE_COLS}

    df = raw.rename(columns={
        "approx_cost(for two people)": "approx_cost",
        "listed_in(type)": "listed_type",
    }).copy()

    report["exact_duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df["rate"] = clean_rate(df["rate"])
    df["approx_cost"] = clean_cost(df["approx_cost"])
    for col in ("online_order", "book_table"):
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)
    df["name"] = df["name"].astype("string").str.strip()
    df["address"] = df["address"].astype("string").str.strip()

    # Merge the per-category listings into one restaurant row.
    grouped = df.groupby(["name", "address"], sort=False)
    rest = grouped.agg(
        online_order=("online_order", "max"),
        book_table=("book_table", "max"),
        rate=("rate", "mean"),
        votes=("votes", "max"),
        location=("location", "first"),
        rest_type=("rest_type", "first"),
        cuisines=("cuisines", "first"),
        approx_cost=("approx_cost", "median"),
        service_type=("listed_type", _primary_service),
        listed_types=("listed_type", lambda v: sorted(set(v))),
    ).reset_index()
    report["unique_restaurants"] = len(rest)
    report["listings_merged_away"] = int(len(df) - len(rest))

    # Missing values.
    missing_before = rest[["rate", "location", "rest_type", "cuisines", "approx_cost"]].isna().sum()
    report["missing_before"] = {k: int(v) for k, v in missing_before.items()}
    rest = rest.dropna(subset=["location", "cuisines"])
    rest["rest_type"] = rest["rest_type"].fillna("Unknown")
    # Fill missing cost with the median cost of the same restaurant type.
    rest["approx_cost"] = rest["approx_cost"].fillna(
        rest.groupby("rest_type")["approx_cost"].transform("median")
    ).fillna(rest["approx_cost"].median())
    rest["approx_cost"] = rest["approx_cost"].astype(float).round(0)

    # Derived attributes and concept hierarchies.
    rest["cuisine_list"] = split_list(rest["cuisines"])
    rest["n_cuisines"] = rest["cuisine_list"].map(len)
    rest["primary_cuisine"] = rest["cuisine_list"].map(lambda c: c[0])
    rest["cuisine_groups"] = rest["cuisine_list"].map(
        lambda cs: sorted({cuisine_to_group(c) for c in cs})
    )
    rest["rest_types"] = split_list(rest["rest_type"])
    rest["primary_type"] = rest["rest_types"].map(lambda t: t[0] if t else "Unknown")
    rest["area"] = rest["location"].map(location_to_area)
    rest["zone"] = rest["location"].map(LOCATION_TO_ZONE).fillna("Other")
    rest["city"] = "Bangalore"

    # Discretisation.
    rest["cost_level"] = pd.cut(
        rest["approx_cost"], bins=config.COST_BINS, labels=config.COST_LABELS
    ).astype(str)

    rated = rest[rest["rate"].notna()].copy()
    unrated = rest[rest["rate"].isna()].copy()
    report["unrated_new_restaurants"] = len(unrated)

    rated["rate"] = rated["rate"].round(2)
    rated["rating_level"] = pd.cut(
        rated["rate"], bins=config.RATING_BINS, labels=config.RATING_LABELS, right=False
    ).astype(str)
    q1, q2 = rated["votes"].quantile([1 / 3, 2 / 3])
    rated["popularity"] = pd.cut(
        rated["votes"], bins=[-1, q1, q2, np.inf], labels=["Unnoticed", "Known", "Popular"]
    ).astype(str)
    report["popularity_vote_cutoffs"] = [float(q1), float(q2)]

    rated["highly_rated"] = (rated["rate"] >= config.HIGH_RATING_THRESHOLD).astype(int)
    rated = rated.reset_index(drop=True)
    rated.insert(0, "restaurant_id", np.arange(1, len(rated) + 1))
    unrated = unrated.reset_index(drop=True)

    report["final_rated_restaurants"] = len(rated)
    report["highly_rated_share"] = round(float(rated["highly_rated"].mean()), 4)
    report["cost_level_counts"] = rated["cost_level"].value_counts().to_dict()
    report["rating_level_counts"] = rated["rating_level"].value_counts().to_dict()
    return rated, unrated, report


def _serialise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in LIST_COLS:
        if col in out:
            out[col] = out[col].map("|".join)
    return out


def save(rated: pd.DataFrame, unrated: pd.DataFrame, report: dict) -> None:
    config.ensure_dirs()
    _serialise(rated).to_csv(config.CLEAN_CSV, index=False)
    _serialise(unrated).to_csv(config.UNRATED_CSV, index=False)
    with open(config.TABLE_DIR / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


def load_clean(path=config.CLEAN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in LIST_COLS:
        if col in df:
            df[col] = df[col].fillna("").map(lambda v: [x for x in str(v).split("|") if x])
    return df


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rated, unrated, report = preprocess(load_raw())
    save(rated, unrated, report)
    return rated, unrated, report
