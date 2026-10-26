"""Step 2 - Data warehouse and OLAP (Module 1).

Star schema stored in SQLite:

                    dim_location (location -> area -> zone -> city)
                          |
    dim_type ------ fact_restaurant ------ dim_service
  (rest type,             |               (online order,
   service type)    dim_cuisine            table booking)
                  (cuisine -> group)
                          |
              bridge_restaurant_cuisine

The fact grain is one restaurant. Its measures are rating, votes and cost.
`fact_restaurant.cuisine_id` points at the primary (first listed) cuisine;
because a restaurant usually serves several cuisines, the bridge table holds
the full many-to-many link for cuisine-level analysis.
"""
import sqlite3
from contextlib import closing

import pandas as pd

from . import config
from .mappings import cuisine_to_group

SCHEMA = """
DROP TABLE IF EXISTS bridge_restaurant_cuisine;
DROP TABLE IF EXISTS fact_restaurant;
DROP TABLE IF EXISTS dim_location;
DROP TABLE IF EXISTS dim_cuisine;
DROP TABLE IF EXISTS dim_type;
DROP TABLE IF EXISTS dim_service;

CREATE TABLE dim_location (
    location_id INTEGER PRIMARY KEY,
    location    TEXT NOT NULL,
    area        TEXT NOT NULL,
    zone        TEXT NOT NULL,
    city        TEXT NOT NULL
);
CREATE TABLE dim_cuisine (
    cuisine_id    INTEGER PRIMARY KEY,
    cuisine       TEXT NOT NULL,
    cuisine_group TEXT NOT NULL
);
CREATE TABLE dim_type (
    type_id      INTEGER PRIMARY KEY,
    rest_type    TEXT NOT NULL,
    service_type TEXT NOT NULL
);
CREATE TABLE dim_service (
    service_id   INTEGER PRIMARY KEY,
    online_order TEXT NOT NULL,
    book_table   TEXT NOT NULL
);
CREATE TABLE fact_restaurant (
    restaurant_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    location_id   INTEGER REFERENCES dim_location(location_id),
    type_id       INTEGER REFERENCES dim_type(type_id),
    service_id    INTEGER REFERENCES dim_service(service_id),
    cuisine_id    INTEGER REFERENCES dim_cuisine(cuisine_id),
    cost_level    TEXT,
    rating_level  TEXT,
    rate          REAL,
    votes         INTEGER,
    approx_cost   REAL,
    highly_rated  INTEGER
);
CREATE TABLE bridge_restaurant_cuisine (
    restaurant_id INTEGER REFERENCES fact_restaurant(restaurant_id),
    cuisine_id    INTEGER REFERENCES dim_cuisine(cuisine_id),
    PRIMARY KEY (restaurant_id, cuisine_id)
);
CREATE INDEX ix_fact_location ON fact_restaurant(location_id);
CREATE INDEX ix_fact_type ON fact_restaurant(type_id);
CREATE INDEX ix_bridge_cuisine ON bridge_restaurant_cuisine(cuisine_id);

-- Convenience view joining the fact table to all its dimensions.
CREATE VIEW IF NOT EXISTS v_restaurant AS
SELECT f.restaurant_id, f.name, l.location, l.area, l.zone, l.city,
       t.rest_type, t.service_type, s.online_order, s.book_table,
       c.cuisine AS primary_cuisine, c.cuisine_group AS primary_cuisine_group,
       f.cost_level, f.rating_level, f.rate, f.votes, f.approx_cost, f.highly_rated
FROM fact_restaurant f
JOIN dim_location l ON f.location_id = l.location_id
JOIN dim_type     t ON f.type_id     = t.type_id
JOIN dim_service  s ON f.service_id  = s.service_id
JOIN dim_cuisine  c ON f.cuisine_id  = c.cuisine_id;
"""

YES_NO = {1: "Yes", 0: "No"}


def _dimension(df: pd.DataFrame, cols: list[str], key: str) -> pd.DataFrame:
    dim = df[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)
    dim.insert(0, key, range(1, len(dim) + 1))
    return dim


def build(df: pd.DataFrame, db_path=config.WAREHOUSE_DB) -> dict:
    """Load the cleaned restaurants into the star schema. Returns table row counts."""
    config.ensure_dirs()
    src = df.copy()
    src["online_order"] = src["online_order"].map(YES_NO)
    src["book_table"] = src["book_table"].map(YES_NO)
    src["rest_type"] = src["primary_type"]

    dim_location = _dimension(src, ["location", "area", "zone", "city"], "location_id")
    dim_type = _dimension(src, ["rest_type", "service_type"], "type_id")
    dim_service = _dimension(src, ["online_order", "book_table"], "service_id")

    bridge = src[["restaurant_id", "cuisine_list"]].explode("cuisine_list")
    bridge = bridge.rename(columns={"cuisine_list": "cuisine"}).drop_duplicates()
    dim_cuisine = pd.DataFrame({"cuisine": sorted(bridge["cuisine"].unique())})
    dim_cuisine["cuisine_group"] = dim_cuisine["cuisine"].map(cuisine_to_group)
    dim_cuisine.insert(0, "cuisine_id", range(1, len(dim_cuisine) + 1))

    fact = (
        src.merge(dim_location, on=["location", "area", "zone", "city"])
        .merge(dim_type, on=["rest_type", "service_type"])
        .merge(dim_service, on=["online_order", "book_table"])
        .merge(dim_cuisine[["cuisine_id", "cuisine"]], left_on="primary_cuisine", right_on="cuisine")
    )
    fact = fact[[
        "restaurant_id", "name", "location_id", "type_id", "service_id", "cuisine_id",
        "cost_level", "rating_level", "rate", "votes", "approx_cost", "highly_rated",
    ]].sort_values("restaurant_id")
    bridge = bridge.merge(dim_cuisine[["cuisine_id", "cuisine"]], on="cuisine")[
        ["restaurant_id", "cuisine_id"]
    ]

    with closing(sqlite3.connect(db_path)) as con:
        con.executescript("DROP VIEW IF EXISTS v_restaurant;" + SCHEMA)
        tables = {
            "dim_location": dim_location, "dim_cuisine": dim_cuisine,
            "dim_type": dim_type, "dim_service": dim_service,
            "fact_restaurant": fact, "bridge_restaurant_cuisine": bridge,
        }
        for name, table in tables.items():
            table.to_sql(name, con, if_exists="append", index=False)
        con.commit()
    return {name: len(t) for name, t in tables.items()}


def query(sql: str, params: tuple = (), db_path=config.WAREHOUSE_DB) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as con:
        return pd.read_sql_query(sql, con, params=params)


# ---------------------------------------------------------------- OLAP --------
MEASURES = """COUNT(*) AS restaurants,
       ROUND(AVG(f.rate), 2) AS avg_rating,
       ROUND(AVG(f.votes), 0) AS avg_votes,
       ROUND(AVG(f.approx_cost), 0) AS avg_cost,
       ROUND(100.0 * AVG(f.highly_rated), 1) AS pct_highly_rated"""

GEO_LEVELS = ["location", "area", "zone", "city"]


def rollup(level: str = "zone") -> pd.DataFrame:
    """Aggregate rating measures at one level of the geography hierarchy.

    Rolling up = moving from `location` towards `city` (fewer, bigger groups);
    drilling down = the reverse.
    """
    if level not in GEO_LEVELS:
        raise ValueError(f"level must be one of {GEO_LEVELS}")
    return query(f"""
        SELECT l.{level}, {MEASURES}
        FROM fact_restaurant f JOIN dim_location l USING (location_id)
        GROUP BY l.{level} ORDER BY restaurants DESC""")


def drill_down(zone: str) -> pd.DataFrame:
    """Drill from one zone down to its individual locations."""
    return query(f"""
        SELECT l.area, l.location, {MEASURES}
        FROM fact_restaurant f JOIN dim_location l USING (location_id)
        WHERE l.zone = ?
        GROUP BY l.area, l.location ORDER BY restaurants DESC""", (zone,))


def slice_(rest_type: str = "Cafe") -> pd.DataFrame:
    """Slice: fix one dimension value (restaurant type) and look at the rest by zone."""
    return query(f"""
        SELECT l.zone, {MEASURES}
        FROM fact_restaurant f
        JOIN dim_location l USING (location_id)
        JOIN dim_type t USING (type_id)
        WHERE t.rest_type = ?
        GROUP BY l.zone ORDER BY avg_rating DESC""", (rest_type,))


def dice(rest_type: str = "Cafe", area: str = "Koramangala", cost_level: str = "Low") -> pd.DataFrame:
    """Dice: fix several dimensions at once, returns the matching restaurants."""
    return query("""
        SELECT f.name, l.location, f.rate, f.votes, f.approx_cost, s.online_order, s.book_table
        FROM fact_restaurant f
        JOIN dim_location l USING (location_id)
        JOIN dim_type t USING (type_id)
        JOIN dim_service s USING (service_id)
        WHERE t.rest_type = ? AND l.area = ? AND f.cost_level = ?
        ORDER BY f.rate DESC, f.votes DESC""", (rest_type, area, cost_level))


def pivot(rows: str = "zone", cols: str = "cost_level", value: str = "avg_rating") -> pd.DataFrame:
    """Pivot (rotate) a two-dimensional cube face: rows x cols -> measure."""
    sql_cols = {
        "zone": "l.zone", "area": "l.area", "cost_level": "f.cost_level",
        "rest_type": "t.rest_type", "service_type": "t.service_type",
        "online_order": "s.online_order", "book_table": "s.book_table",
    }
    cube = query(f"""
        SELECT {sql_cols[rows]} AS r, {sql_cols[cols]} AS c, {MEASURES}
        FROM fact_restaurant f
        JOIN dim_location l USING (location_id)
        JOIN dim_type t USING (type_id)
        JOIN dim_service s USING (service_id)
        GROUP BY r, c""")
    table = cube.pivot(index="r", columns="c", values=value)
    table.index.name, table.columns.name = rows, cols
    if cols == "cost_level":
        table = table[[c for c in config.COST_LABELS if c in table.columns]]
    return table


def services_effect() -> pd.DataFrame:
    """Does offering online ordering / table booking go with better ratings?"""
    return query(f"""
        SELECT s.online_order, s.book_table, {MEASURES}
        FROM fact_restaurant f JOIN dim_service s USING (service_id)
        GROUP BY s.online_order, s.book_table""")


def cuisine_by_area(area: str | None = None, min_restaurants: int = 1) -> pd.DataFrame:
    """Cuisine-level measures through the bridge table, optionally sliced by area."""
    where = "WHERE l.area = ?" if area else ""
    params = (area, min_restaurants) if area else (min_restaurants,)
    return query(f"""
        SELECT c.cuisine, c.cuisine_group, {MEASURES}
        FROM bridge_restaurant_cuisine b
        JOIN fact_restaurant f USING (restaurant_id)
        JOIN dim_cuisine c USING (cuisine_id)
        JOIN dim_location l USING (location_id)
        {where}
        GROUP BY c.cuisine, c.cuisine_group
        HAVING COUNT(*) >= ?
        ORDER BY restaurants DESC""", params)


def market_gaps(min_area_restaurants: int = 40, top: int = 25) -> pd.DataFrame:
    """Find (area, cuisine group) cells where demand is high but supply is low.

    demand_index = (avg votes of the group in the area / avg votes of the area)
                   / (avg votes of the group city-wide / avg votes city-wide)
    supply_index = share of the area's restaurants serving that group
                   / share of the whole city's restaurants serving it
    gap_score    = demand_index / supply_index

    Both indexes are relative to the city norm (1.0), so a cuisine that is
    popular everywhere does not dominate. A score well above 1 means people in
    that area engage with the cuisine more than usual, yet fewer restaurants
    offer it there than the city norm.
    """
    cells = query("""
        SELECT l.area, c.cuisine_group,
               COUNT(DISTINCT f.restaurant_id) AS restaurants,
               AVG(f.votes) AS avg_votes, AVG(f.rate) AS avg_rating
        FROM bridge_restaurant_cuisine b
        JOIN fact_restaurant f USING (restaurant_id)
        JOIN dim_cuisine c USING (cuisine_id)
        JOIN dim_location l USING (location_id)
        GROUP BY l.area, c.cuisine_group""")
    area_tot = query("""
        SELECT l.area, COUNT(*) AS area_restaurants, AVG(f.votes) AS area_avg_votes
        FROM fact_restaurant f JOIN dim_location l USING (location_id)
        GROUP BY l.area""")
    city = query("""
        SELECT c.cuisine_group, COUNT(DISTINCT f.restaurant_id) AS restaurants,
               AVG(f.votes) AS group_avg_votes
        FROM bridge_restaurant_cuisine b
        JOIN fact_restaurant f USING (restaurant_id)
        JOIN dim_cuisine c USING (cuisine_id)
        GROUP BY c.cuisine_group""").set_index("cuisine_group")
    total = area_tot["area_restaurants"].sum()
    city_avg_votes = query("SELECT AVG(votes) AS v FROM fact_restaurant")["v"].iloc[0]

    g = cells.merge(area_tot, on="area")
    g = g[(g["area_restaurants"] >= min_area_restaurants) & (g["restaurants"] >= 3)].copy()
    g["area_share"] = g["restaurants"] / g["area_restaurants"]
    g["city_share"] = g["cuisine_group"].map(city["restaurants"] / total)
    group_lift = g["cuisine_group"].map(city["group_avg_votes"] / city_avg_votes)
    g["demand_index"] = (g["avg_votes"] / g["area_avg_votes"]) / group_lift
    g["supply_index"] = g["area_share"] / g["city_share"]
    g["gap_score"] = g["demand_index"] / g["supply_index"]
    g = g[(g["demand_index"] > 1) & (g["supply_index"] < 1)]
    cols = ["area", "cuisine_group", "restaurants", "area_restaurants", "avg_rating",
            "avg_votes", "demand_index", "supply_index", "gap_score"]
    return g.sort_values("gap_score", ascending=False)[cols].head(top).round(2).reset_index(drop=True)


def run_olap_demo() -> dict[str, pd.DataFrame]:
    """The OLAP operations shown in the report, saved to outputs/tables."""
    results = {
        "olap_rollup_location": rollup("location"),
        "olap_rollup_area": rollup("area"),
        "olap_rollup_zone": rollup("zone"),
        "olap_rollup_city": rollup("city"),
        "olap_drilldown_south": drill_down("South"),
        "olap_slice_cafe": slice_("Cafe"),
        "olap_dice_cafe_koramangala_low": dice("Cafe", "Koramangala", "Low"),
        "olap_pivot_zone_cost": pivot("zone", "cost_level").reset_index(),
        "olap_pivot_type_cost": pivot("rest_type", "cost_level").reset_index(),
        "olap_services_effect": services_effect(),
        "olap_cuisines_city": cuisine_by_area(min_restaurants=30),
        "market_gaps": market_gaps(),
    }
    for name, table in results.items():
        table.to_csv(config.TABLE_DIR / f"{name}.csv", index=False)
    return results
