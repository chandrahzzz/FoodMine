"""Step 5 - Cluster analysis and outliers (Module 6).

Restaurants are grouped on three measures: rating, log(votes) and log(cost),
standardised so each counts equally.

* K-Means          - k chosen with the elbow (SSE) and silhouette score
* K-Medoids        - CLARA: PAM (build + swap) run on several random samples,
                     keeping the medoids with the lowest cost on the full data.
                     Written here because scikit-learn-extra does not build
                     against NumPy 2.
* DBSCAN           - eps picked at the knee of the sorted k-distance curve;
                     points labelled noise are treated as outliers
* Rule-based outliers named in the proposal: expensive but poorly rated,
  and cheap with an extreme number of votes
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config

FEATURES = ["rate", "log_votes", "log_cost"]


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    raw = pd.DataFrame({
        "rate": df["rate"],
        "log_votes": np.log1p(df["votes"]),
        "log_cost": np.log1p(df["approx_cost"]),
    })
    scaler = StandardScaler().fit(raw)
    return scaler.transform(raw), scaler


# ---------------------------------------------------------------- K-Means ----
def choose_k(X: np.ndarray, ks=range(2, 10)) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
        rows.append({"k": k, "sse": km.inertia_,
                     "silhouette": silhouette_score(X[sample], km.labels_[sample])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- K-Medoids --
def _pam(D: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    """PAM on a precomputed distance matrix: greedy BUILD then SWAP until no
    swap lowers the total distance. Returns medoid indices into D."""
    n = len(D)
    medoids = [int(np.argmin(D.sum(axis=1)))]
    for _ in range(1, k):
        nearest = D[:, medoids].min(axis=1)
        gain = np.maximum(nearest[None, :] - D, 0).sum(axis=1)
        gain[medoids] = -1
        medoids.append(int(np.argmax(gain)))
    medoids = np.array(medoids)

    cost = D[:, medoids].min(axis=1).sum()
    for _ in range(max_iter):
        best = (0.0, None, None)
        for mi in range(k):
            others = np.delete(medoids, mi)
            nearest_other = D[:, others].min(axis=1) if k > 1 else np.full(n, np.inf)
            # cost of replacing medoid mi with every candidate h, vectorised over h
            new_costs = np.minimum(nearest_other[:, None], D).sum(axis=0)
            new_costs[medoids] = np.inf
            h = int(np.argmin(new_costs))
            delta = new_costs[h] - cost
            if delta < best[0]:
                best = (delta, mi, h)
        if best[1] is None:
            break
        medoids[best[1]] = best[2]
        cost += best[0]
    return medoids


class CLARA:
    """K-Medoids for large data (Kaufman & Rousseeuw): PAM on samples."""

    def __init__(self, n_clusters: int, n_samples: int = 5, sample_size: int = 800,
                 random_state: int = config.RANDOM_STATE):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "CLARA":
        rng = np.random.default_rng(self.random_state)
        best_cost, best_medoids = np.inf, None
        for _ in range(self.n_samples):
            idx = rng.choice(len(X), size=min(self.sample_size, len(X)), replace=False)
            S = X[idx]
            D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
            medoids = idx[_pam(D, self.n_clusters)]
            cost = self._distances(X, X[medoids]).min(axis=1).sum()
            if cost < best_cost:
                best_cost, best_medoids = cost, medoids
        self.medoid_indices_ = best_medoids
        self.cluster_centers_ = X[best_medoids]
        self.labels_ = self.predict(X)
        self.inertia_ = best_cost
        return self

    @staticmethod
    def _distances(X, C):
        return np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))

    def predict(self, X):
        return self._distances(X, self.cluster_centers_).argmin(axis=1)


# ---------------------------------------------------------------- DBSCAN -----
def knee_eps(X: np.ndarray, min_samples: int) -> tuple[float, np.ndarray]:
    """Distance to the min_samples-th neighbour, sorted; eps = point farthest
    from the straight line joining the curve's ends (the 'knee')."""
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    kd = np.sort(dist[:, -1])
    x = np.linspace(0, 1, len(kd))
    y = (kd - kd.min()) / (kd.max() - kd.min())
    knee = int(np.argmax(np.abs(y - x)))
    return float(kd[knee]), kd


# ---------------------------------------------------------------- naming -----
def name_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Name each cluster from its median cost, rating and votes vs. the city."""
    names = {}
    cost_q = df["approx_cost"].quantile([0.4, 0.8]).to_numpy()
    votes_q = df["votes"].quantile([0.4, 0.75]).to_numpy()
    for c in sorted(set(labels) - {-1}):
        g = df[labels == c]
        cost, rate, votes = g["approx_cost"].median(), g["rate"].median(), g["votes"].median()
        price = "Budget" if cost <= cost_q[0] else ("Mid-range" if cost <= cost_q[1] else "Premium")
        rating = "highly rated" if rate >= 4.0 else ("average" if rate >= 3.6 else "poorly rated")
        reach = "unnoticed" if votes <= votes_q[0] else ("known" if votes <= votes_q[1] else "popular")
        names[c] = f"{price}, {rating}, {reach}"
    # Keep names unique if two clusters land on the same description.
    seen: dict[str, int] = {}
    for c, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({seen[n]})"
    return names


def profile(df: pd.DataFrame, labels: np.ndarray, names: dict) -> pd.DataFrame:
    g = df.assign(cluster=labels)
    g = g[g["cluster"] != -1]
    prof = g.groupby("cluster").agg(
        restaurants=("restaurant_id", "count"),
        median_rating=("rate", "median"),
        median_votes=("votes", "median"),
        median_cost=("approx_cost", "median"),
        pct_online_order=("online_order", "mean"),
        pct_table_booking=("book_table", "mean"),
        pct_highly_rated=("highly_rated", "mean"),
        top_type=("primary_type", lambda s: s.mode().iloc[0]),
    ).reset_index()
    for c in ("pct_online_order", "pct_table_booking", "pct_highly_rated"):
        prof[c] = (prof[c] * 100).round(1)
    prof.insert(1, "name", prof["cluster"].map(names))
    return prof


# ---------------------------------------------------------------- run --------
def run(df: pd.DataFrame) -> dict:
    X, _ = feature_matrix(df)
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample = rng.choice(len(X), size=min(4000, len(X)), replace=False)

    k_table = choose_k(X)
    # Best silhouette among k = 3..6 keeps the groups few enough to describe.
    candidates = k_table[(k_table["k"] >= 3) & (k_table["k"] <= 6)]
    k = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])

    km = KMeans(k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
    kmed = CLARA(k).fit(X)

    min_samples = 2 * X.shape[1]
    eps, kdist = knee_eps(X, min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    km_names = name_clusters(df, km.labels_)
    kmed_names = name_clusters(df, kmed.labels_)
    db_names = name_clusters(df, db.labels_)

    comparison = pd.DataFrame([
        {"method": "K-Means", "clusters": k, "noise_points": 0,
         "silhouette": silhouette_score(X[sample], km.labels_[sample]),
         "sse_or_cost": km.inertia_},
        {"method": "K-Medoids (CLARA)", "clusters": k, "noise_points": 0,
         "silhouette": silhouette_score(X[sample], kmed.labels_[sample]),
         "sse_or_cost": float(((X - kmed.cluster_centers_[kmed.labels_]) ** 2).sum())},
        {"method": "DBSCAN", "clusters": int(len(set(db.labels_)) - (-1 in db.labels_)),
         "noise_points": int((db.labels_ == -1).sum()),
         "silhouette": (silhouette_score(X[sample][db.labels_[sample] != -1],
                                         db.labels_[sample][db.labels_[sample] != -1])
                        if len(set(db.labels_) - {-1}) > 1 else np.nan),
         "sse_or_cost": np.nan},
    ]).round(4)
    agreement = {
        "ARI K-Means vs K-Medoids": round(adjusted_rand_score(km.labels_, kmed.labels_), 4),
        "ARI K-Means vs DBSCAN": round(adjusted_rand_score(km.labels_, db.labels_), 4),
    }
    medoids = df.iloc[kmed.medoid_indices_][["name", "location", "primary_type", "rate", "votes", "approx_cost"]]
    medoids = medoids.assign(cluster=range(k), cluster_name=[kmed_names[i] for i in range(k)])

    assigned = df[["restaurant_id", "name", "location", "area", "zone", "primary_type", "cuisines",
                   "rate", "votes", "approx_cost", "online_order", "book_table", "highly_rated"]].copy()
    assigned["kmeans_cluster"] = km.labels_
    assigned["kmeans_name"] = assigned["kmeans_cluster"].map(km_names)
    assigned["kmedoids_cluster"] = kmed.labels_
    assigned["dbscan_cluster"] = db.labels_

    # Outliers: DBSCAN noise + the two rule-based kinds from the proposal.
    cost_hi, cost_lo = df["approx_cost"].quantile(0.9), df["approx_cost"].quantile(0.25)
    votes_hi = df["votes"].quantile(0.99)
    reasons = pd.Series([[] for _ in range(len(df))], index=df.index)
    for mask, why in (
        (db.labels_ == -1, "DBSCAN noise"),
        ((df["approx_cost"] >= cost_hi) & (df["rate"] < 3.5), "Expensive but poorly rated"),
        ((df["approx_cost"] <= cost_lo) & (df["votes"] >= votes_hi), "Cheap with extreme votes"),
    ):
        for i in np.flatnonzero(np.asarray(mask)):
            reasons.iloc[i].append(why)
    assigned["outlier_reasons"] = reasons.map("; ".join).to_numpy()
    outliers = assigned[assigned["outlier_reasons"] != ""].sort_values("votes", ascending=False)

    out = {
        "cluster_k_selection": k_table.round(4),
        "cluster_profile_kmeans": profile(df, km.labels_, km_names),
        "cluster_profile_kmedoids": profile(df, kmed.labels_, kmed_names),
        "cluster_profile_dbscan": profile(df, db.labels_, db_names),
        "cluster_comparison": comparison,
        "kmedoids_medoids": medoids.reset_index(drop=True),
        "cluster_assignments": assigned,
        "outliers": outliers.reset_index(drop=True),
    }
    for name, table in out.items():
        table.to_csv(config.TABLE_DIR / f"{name}.csv", index=False)
    with open(config.TABLE_DIR / "cluster_details.json", "w") as f:
        json.dump({"k": k, "dbscan_eps": round(eps, 4), "dbscan_min_samples": min_samples,
                   "outlier_rules": {"expensive_cost_at_least": float(cost_hi),
                                     "cheap_cost_at_most": float(cost_lo),
                                     "extreme_votes_at_least": float(votes_hi)},
                   **agreement}, f, indent=2)
    out.update({"X": X, "kdist": kdist, "eps": eps, "k": k, "agreement": agreement,
                "kmeans_names": km_names})
    return out
