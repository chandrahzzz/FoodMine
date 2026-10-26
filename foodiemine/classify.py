"""Step 4 - Classification (Module 5): will a restaurant be Highly Rated (>= 4.0)?

Only features an owner knows *before* customers start rating are used
(services, location, type, cost, cuisines). Votes and rating are left out,
so the same model can score brand-new restaurants that have no rating yet.

Models
* Naive Bayes (CategoricalNB) - plain and with balanced class priors
* K-Nearest Neighbours (lazy learner) - k tuned by cross-validation,
  plain and with random oversampling of the minority class
* Bagging ensembles of KNN and of decision trees
* Bayesian Belief Network (pgmpy) - expert-drawn structure, plus a
  structure learned by hill-climbing for comparison
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config
from .mappings import CUISINE_GROUPS, cuisine_to_group

CAT_COLS = ["online_order", "book_table", "location", "primary_type", "service_type", "cost_level"]
TOP_N_CUISINES = 30
TARGET = "highly_rated"
N_JOBS = 2  # keep memory use modest on laptops


# ---------------------------------------------------------------- features ----
def feature_spec(df: pd.DataFrame) -> dict:
    """Category lists fixed from the full dataset so every split shares them."""
    top = df["cuisine_list"].explode().value_counts().head(TOP_N_CUISINES).index.tolist()
    return {
        "categories": {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT_COLS},
        "top_cuisines": top,
        "cuisine_groups": sorted(CUISINE_GROUPS) + ["Other"],
    }


def make_features(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Raw restaurant rows -> model input table.

    Needs columns: online_order, book_table, location, primary_type,
    service_type, approx_cost, cuisine_list.
    """
    X = pd.DataFrame(index=df.index)
    X["online_order"] = df["online_order"].map(lambda v: "Yes" if int(v) else "No")
    X["book_table"] = df["book_table"].map(lambda v: "Yes" if int(v) else "No")
    for c in ("location", "primary_type", "service_type"):
        X[c] = df[c].astype(str)
    X["cost_level"] = pd.cut(df["approx_cost"], config.COST_BINS, labels=config.COST_LABELS).astype(str)
    X["approx_cost"] = df["approx_cost"].astype(float)
    X["n_cuisines"] = df["cuisine_list"].map(len).clip(upper=8)
    for c in spec["top_cuisines"]:
        X[f"cz_{c}"] = df["cuisine_list"].map(lambda cs, c=c: int(c in cs))
    groups = df["cuisine_list"].map(lambda cs: {cuisine_to_group(c) for c in cs})
    for g in spec["cuisine_groups"]:
        X[f"cg_{g}"] = groups.map(lambda gs, g=g: int(g in gs))
    # Map categories never seen in the spec (only possible for user input) to the most common one.
    for c in CAT_COLS:
        known = spec["categories"][c]
        X.loc[~X[c].isin(known), c] = known[0]
    return X


def _flag_cols(spec: dict) -> list[str]:
    return [f"cz_{c}" for c in spec["top_cuisines"]] + [f"cg_{g}" for g in spec["cuisine_groups"]]


def _log1p(x):
    return np.log1p(x)


def nb_pipeline(spec: dict, balanced: bool = False) -> Pipeline:
    cats = [spec["categories"][c] for c in CAT_COLS]
    flags = _flag_cols(spec)
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(categories=cats), CAT_COLS),
        ("ncuis", "passthrough", ["n_cuisines"]),
        ("flags", "passthrough", flags),
    ])
    min_categories = [len(c) for c in cats] + [9] + [2] * len(flags)
    nb = CategoricalNB(alpha=1.0, min_categories=min_categories,
                       class_prior=[0.5, 0.5] if balanced else None)
    return Pipeline([("prep", pre), ("model", nb)])


def onehot_preprocessor(spec: dict) -> ColumnTransformer:
    cats = [spec["categories"][c] for c in CAT_COLS]
    return ColumnTransformer([
        ("cat", OneHotEncoder(categories=cats, handle_unknown="ignore"), CAT_COLS),
        ("num", Pipeline([("log", FunctionTransformer(_log1p)), ("scale", StandardScaler())]),
         ["approx_cost", "n_cuisines"]),
        ("flags", "passthrough", _flag_cols(spec)),
    ])


class RandomOversampler(BaseEstimator, ClassifierMixin):
    """Wraps a classifier; duplicates minority-class rows at fit time until classes are balanced."""

    def __init__(self, estimator=None, random_state=config.RANDOM_STATE):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        idx = [np.flatnonzero(y == c) for c in classes]
        top = counts.max()
        sample = np.concatenate([
            np.concatenate([i, rng.choice(i, top - len(i), replace=True)]) for i in idx
        ])
        Xs = X.iloc[sample] if hasattr(X, "iloc") else X[sample]
        self.estimator_ = clone(self.estimator).fit(Xs, y[sample])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def tune_knn(spec: dict, X: pd.DataFrame, y: pd.Series) -> tuple[int, str, pd.DataFrame]:
    pipe = Pipeline([("prep", onehot_preprocessor(spec)), ("model", KNeighborsClassifier())])
    grid = GridSearchCV(
        pipe,
        {"model__n_neighbors": [5, 9, 15, 21, 31, 45], "model__weights": ["uniform", "distance"]},
        scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE), n_jobs=N_JOBS,
    )
    grid.fit(X, y)
    res = pd.DataFrame(grid.cv_results_)[["param_model__n_neighbors", "param_model__weights", "mean_test_score"]]
    res.columns = ["k", "weights", "cv_f1"]
    best = grid.best_params_
    return best["model__n_neighbors"], best["model__weights"], res


def build_models(spec: dict, k: int, weights: str) -> dict:
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
    rs = config.RANDOM_STATE
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": nb_pipeline(spec),
        "Naive Bayes (balanced priors)": nb_pipeline(spec, balanced=True),
        f"KNN (k={k})": Pipeline([("prep", onehot_preprocessor(spec)), ("model", knn)]),
        f"KNN (k={k}) + oversampling": Pipeline([
            ("prep", onehot_preprocessor(spec)), ("model", RandomOversampler(knn)),
        ]),
        "Bagging (KNN)": Pipeline([
            ("prep", onehot_preprocessor(spec)),
            ("model", BaggingClassifier(knn, n_estimators=15, max_samples=0.8, random_state=rs, n_jobs=N_JOBS)),
        ]),
        "Bagging (decision trees)": Pipeline([
            ("prep", onehot_preprocessor(spec)),
            ("model", BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=3, random_state=rs),
                                        n_estimators=100, random_state=rs, n_jobs=N_JOBS)),
        ]),
    }


def evaluate(models: dict, X_tr, y_tr, X_te, y_te) -> tuple[pd.DataFrame, dict, dict, dict]:
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    rows, confusions, rocs, fitted = [], {}, {}, {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1"], n_jobs=1)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]
        fitted[name] = model
        confusions[name] = confusion_matrix(y_te, pred).tolist()
        if name != "Majority baseline":
            fpr, tpr, _ = roc_curve(y_te, proba)
            rocs[name] = (fpr, tpr)
        rows.append({
            "model": name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_f1": scores["test_f1"].mean(),
            "test_accuracy": accuracy_score(y_te, pred),
            "test_precision": precision_score(y_te, pred, zero_division=0),
            "test_recall": recall_score(y_te, pred, zero_division=0),
            "test_f1": f1_score(y_te, pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_te, proba),
        })
    return pd.DataFrame(rows).round(4), confusions, rocs, fitted


def nb_top_signals(nb: Pipeline, spec: dict, top: int = 20) -> pd.DataFrame:
    """Read the fitted Naive Bayes likelihoods: which feature values most raise
    P(highly rated) relative to P(not highly rated)."""
    model = nb.named_steps["model"]
    names = CAT_COLS + ["n_cuisines"] + _flag_cols(spec)
    rows = []
    for j, name in enumerate(names):
        logp = model.feature_log_prob_[j]  # shape (2 classes, n_values)
        if name in spec["categories"]:
            values = spec["categories"][name]
        elif name == "n_cuisines":
            values = [str(v) for v in range(9)]
        else:
            values = ["No", "Yes"]
        counts = model.category_count_[j].sum(axis=0)
        for v_idx, value in enumerate(values[: logp.shape[1]]):
            if counts[v_idx] < 30 or (name.startswith(("cz_", "cg_")) and value == "No"):
                continue
            label = name.replace("cz_", "Cuisine: ").replace("cg_", "Cuisine group: ")
            rows.append({
                "feature": label if value == "Yes" and name.startswith(("cz_", "cg_")) else f"{label} = {value}",
                "likelihood_ratio": float(np.exp(logp[1, v_idx] - logp[0, v_idx])),
                "restaurants": int(counts[v_idx]),
            })
    out = pd.DataFrame(rows).sort_values("likelihood_ratio", ascending=False)
    return pd.concat([out.head(top), out.tail(top)]).drop_duplicates().round(3).reset_index(drop=True)


# ---------------------------------------------------------------- BBN ---------
BBN_EDGES = [
    ("Type", "Cost"), ("Type", "OnlineOrder"), ("Cost", "TableBooking"),
    ("Cost", "Rating"), ("TableBooking", "Rating"), ("OnlineOrder", "Rating"),
    ("Rating", "Popularity"),
]


def bbn_frame(df: pd.DataFrame) -> pd.DataFrame:
    main_types = {"Quick Bites", "Casual Dining", "Cafe", "Delivery", "Dessert Parlor"}
    return pd.DataFrame({
        "Type": df["primary_type"].where(df["primary_type"].isin(main_types), "Other"),
        "Cost": df["cost_level"],
        "OnlineOrder": df["online_order"].map({1: "Yes", 0: "No"}),
        "TableBooking": df["book_table"].map({1: "Yes", 0: "No"}),
        "Rating": df["highly_rated"].map({1: "High", 0: "NotHigh"}),
        "Popularity": df["popularity"],
    }).astype(str)


def run_bbn(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    import logging
    logging.getLogger("pgmpy").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="pgmpy")
    from pgmpy.causal_discovery import HillClimbSearch
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.parameter_estimator import DiscreteBayesianEstimator

    tr, te = bbn_frame(train), bbn_frame(test)
    evidence_cols = ["Type", "Cost", "OnlineOrder", "TableBooking"]

    def fit(edges):
        model = DiscreteBayesianNetwork(edges)
        model = model.fit(tr, estimator=DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=10))
        return model

    def score(model):
        probs = model.predict_probability(te[evidence_cols])
        p_high = probs["Rating_High"].to_numpy()
        y = (te["Rating"] == "High").astype(int).to_numpy()
        pred = (p_high >= 0.5).astype(int)
        return {
            "accuracy": round(accuracy_score(y, pred), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, p_high), 4),
        }

    expert = fit(BBN_EDGES)
    search = HillClimbSearch(scoring_method="bic-d", return_type="dag", show_progress=False).fit(tr)
    learned_edges = list(search.causal_graph_.edges())
    learned = fit(learned_edges)

    infer = VariableElimination(expert)
    queries = []
    for cost in config.COST_LABELS:
        for book in ("No", "Yes"):
            for online in ("No", "Yes"):
                q = infer.query(["Rating"], evidence={"Cost": cost, "TableBooking": book,
                                                      "OnlineOrder": online}, show_progress=False)
                p = dict(zip(q.state_names["Rating"], q.values))
                n = int(((tr["Cost"] == cost) & (tr["TableBooking"] == book)
                         & (tr["OnlineOrder"] == online)).sum())
                queries.append({"Cost": cost, "TableBooking": book, "OnlineOrder": online,
                                "P(Highly rated)": round(float(p["High"]), 3),
                                "training_restaurants": n})
    # Diagnostic reasoning: from the effect (rating) back to a cause (cost).
    diag = infer.query(["Cost"], evidence={"Rating": "High"}, show_progress=False)
    prior = infer.query(["Cost"], show_progress=False)
    diagnostic = pd.DataFrame({
        "Cost": diag.state_names["Cost"],
        "P(Cost)": np.round(prior.values, 3),
        "P(Cost | Highly rated)": np.round(diag.values, 3),
    })
    return {
        "expert_edges": BBN_EDGES,
        "learned_edges": learned_edges,
        "expert_scores": score(expert),
        "learned_scores": score(learned),
        "queries": pd.DataFrame(queries),
        "diagnostic": diagnostic,
        "model": expert,
    }


# ---------------------------------------------------------------- run ---------
def run(df: pd.DataFrame, unrated: pd.DataFrame | None = None) -> dict:
    spec = feature_spec(pd.concat([df, unrated]) if unrated is not None else df)
    X = make_features(df, spec)
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    k, weights, knn_grid = tune_knn(spec, X_tr, y_tr)
    models = build_models(spec, k, weights)
    metrics, confusions, rocs, fitted = evaluate(models, X_tr, y_tr, X_te, y_te)

    # Pick the model with the best cross-validated F1 (never the test set).
    best_name = metrics.loc[metrics["cv_f1"].idxmax(), "model"]
    best = clone(models[best_name]).fit(X, y)  # refit on all rated data for deployment
    nb_full = nb_pipeline(spec).fit(X, y)
    signals = nb_top_signals(nb_full, spec)

    bbn = run_bbn(df.loc[idx_tr], df.loc[idx_te])
    bbn_row = {"model": "Bayesian Belief Network (expert)", "test_accuracy": bbn["expert_scores"]["accuracy"],
               "test_f1": bbn["expert_scores"]["f1"], "test_roc_auc": bbn["expert_scores"]["roc_auc"]}
    bbn_row2 = {"model": "Bayesian Belief Network (hill-climb)", "test_accuracy": bbn["learned_scores"]["accuracy"],
                "test_f1": bbn["learned_scores"]["f1"], "test_roc_auc": bbn["learned_scores"]["roc_auc"]}
    metrics_all = pd.concat([metrics, pd.DataFrame([bbn_row, bbn_row2])], ignore_index=True)

    joblib.dump({"model": best, "name": best_name, "spec": spec, "nb": nb_full,
                 "threshold": config.HIGH_RATING_THRESHOLD},
                config.MODEL_DIR / "rating_model.joblib")
    joblib.dump(bbn["model"], config.MODEL_DIR / "bbn.joblib")

    new_preds = None
    if unrated is not None and len(unrated):
        Xu = make_features(unrated, spec)
        new_preds = unrated[["name", "location", "primary_type", "approx_cost", "cuisines",
                             "online_order", "book_table"]].copy()
        new_preds["p_highly_rated"] = best.predict_proba(Xu)[:, 1].round(3)
        new_preds = new_preds.sort_values("p_highly_rated", ascending=False).reset_index(drop=True)
        new_preds.to_csv(config.TABLE_DIR / "new_restaurant_predictions.csv", index=False)

    metrics_all.to_csv(config.TABLE_DIR / "classification_metrics.csv", index=False)
    knn_grid.to_csv(config.TABLE_DIR / "knn_tuning.csv", index=False)
    signals.to_csv(config.TABLE_DIR / "naive_bayes_signals.csv", index=False)
    bbn["queries"].to_csv(config.TABLE_DIR / "bbn_queries.csv", index=False)
    bbn["diagnostic"].to_csv(config.TABLE_DIR / "bbn_diagnostic.csv", index=False)
    with open(config.TABLE_DIR / "classification_details.json", "w") as f:
        json.dump({
            "best_model": best_name, "knn_k": k, "knn_weights": weights,
            "train_size": len(X_tr), "test_size": len(X_te),
            "positive_share": round(float(y.mean()), 4),
            "confusion_matrices": confusions,
            "bbn_expert_edges": bbn["expert_edges"],
            "bbn_learned_edges": [list(e) for e in bbn["learned_edges"]],
        }, f, indent=2)
    return {"metrics": metrics_all, "confusions": confusions, "rocs": rocs, "best_name": best_name,
            "knn_grid": knn_grid, "signals": signals, "bbn": bbn, "new_predictions": new_preds,
            "spec": spec, "k": k}


def load_model() -> dict:
    return joblib.load(config.MODEL_DIR / "rating_model.joblib")
