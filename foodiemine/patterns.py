"""Step 3 - Frequent pattern mining (Module 4).

Each restaurant becomes a transaction of items, for example
{Cuisine=North Indian, Cuisine=Chinese, Cost=Medium, OnlineOrder=Yes, Rating=Good}.

Three miners find the same frequent itemsets in different ways:
* Apriori    (mlxtend)  - level-wise candidate generation + pruning
* FP-Growth  (mlxtend)  - FP-tree, no candidate generation
* Eclat      (own code) - vertical data format: item -> set of transaction ids,
                          support = size of the tidset intersection
"""
import time

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from . import config


def build_transactions(df: pd.DataFrame, include_zone: bool = True) -> list[list[str]]:
    transactions = []
    for row in df.itertuples(index=False):
        items = [f"Cuisine={c}" for c in row.cuisine_list]
        items += [
            f"Cost={row.cost_level}",
            f"OnlineOrder={'Yes' if row.online_order else 'No'}",
            f"TableBooking={'Yes' if row.book_table else 'No'}",
            f"Type={row.primary_type}",
            f"Service={row.service_type}",
            f"Popularity={row.popularity}",
            f"Rating={row.rating_level}",
        ]
        if include_zone:
            items.append(f"Zone={row.zone}")
        transactions.append(sorted(set(items)))
    return transactions


def one_hot(transactions: list[list[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)


# ---------------------------------------------------------------- Eclat -------
def eclat(transactions: list[list[str]], min_support: float, max_len: int | None = None) -> pd.DataFrame:
    """Eclat on the vertical data format.

    Tidsets are stored as Python ints used as bitsets, so intersecting two
    tidsets is a single `&` and support counting is `int.bit_count()`.
    Output has the same shape as mlxtend: columns `support` and `itemsets`.
    """
    n = len(transactions)
    min_count = min_support * n
    tidsets: dict[str, int] = {}
    for tid, items in enumerate(transactions):
        bit = 1 << tid
        for item in items:
            tidsets[item] = tidsets.get(item, 0) | bit

    frequent = sorted(
        ((item, tids) for item, tids in tidsets.items() if tids.bit_count() >= min_count),
        key=lambda x: x[0],
    )
    results: list[tuple[float, frozenset]] = []

    def grow(prefix: tuple, candidates: list[tuple[str, int]]) -> None:
        for i, (item, tids) in enumerate(candidates):
            itemset = prefix + (item,)
            results.append((tids.bit_count() / n, frozenset(itemset)))
            if max_len is not None and len(itemset) >= max_len:
                continue
            extensions = []
            for other, other_tids in candidates[i + 1:]:
                joined = tids & other_tids
                if joined.bit_count() >= min_count:
                    extensions.append((other, joined))
            if extensions:
                grow(itemset, extensions)

    grow((), frequent)
    return pd.DataFrame(results, columns=["support", "itemsets"])


# ---------------------------------------------------------------- compare -----
def compare_algorithms(
    transactions: list[list[str]],
    supports=(0.20, 0.10, 0.05, 0.03, 0.02),
    max_len: int = 4,
) -> pd.DataFrame:
    """Time the three miners at several minimum supports and check they agree."""
    encoded = one_hot(transactions)
    rows = []
    for s in supports:
        timings, found = {}, {}
        for name, fn in (
            ("Apriori", lambda: apriori(encoded, min_support=s, use_colnames=True, max_len=max_len)),
            ("FP-Growth", lambda: fpgrowth(encoded, min_support=s, use_colnames=True, max_len=max_len)),
            ("Eclat", lambda: eclat(transactions, min_support=s, max_len=max_len)),
        ):
            start = time.perf_counter()
            result = fn()
            timings[name] = time.perf_counter() - start
            found[name] = set(result["itemsets"])
        same = found["Apriori"] == found["FP-Growth"] == found["Eclat"]
        for name in timings:
            rows.append({
                "min_support": s, "algorithm": name,
                "seconds": round(timings[name], 4), "itemsets": len(found[name]),
                "all_three_agree": same,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- rules -------
def _fmt(items) -> str:
    return ", ".join(sorted(items))


def mine_rules(
    transactions: list[list[str]],
    min_support: float = 0.01,
    min_confidence: float = 0.5,
    min_lift: float = 1.2,
    max_len: int = 4,
) -> pd.DataFrame:
    itemsets = fpgrowth(one_hot(transactions), min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift].copy()
    rules["antecedent"] = rules["antecedents"].map(_fmt)
    rules["consequent"] = rules["consequents"].map(_fmt)
    rules["rule"] = "{" + rules["antecedent"] + "} -> {" + rules["consequent"] + "}"
    rules["n_antecedent"] = rules["antecedents"].map(len)
    cols = ["rule", "antecedent", "consequent", "n_antecedent", "support", "confidence",
            "lift", "leverage", "conviction"]
    return rules.sort_values(["lift", "confidence"], ascending=False)[cols].reset_index(drop=True)


def rating_rules(rules: pd.DataFrame, level: str = "Good") -> pd.DataFrame:
    """Rules whose only consequent is the rating level, without popularity in the antecedent.

    Popularity (votes) is dropped from the antecedent because votes are given
    by the same customers who give the rating, so it says little an owner can act on.
    """
    target = f"Rating={level}"
    r = rules[(rules["consequent"] == target) & ~rules["antecedent"].str.contains("Popularity=")]
    return prune_redundant(r).reset_index(drop=True)


def prune_redundant(rules: pd.DataFrame) -> pd.DataFrame:
    """Drop a rule when a more general rule (antecedent is a strict subset, same
    consequent) already reaches at least the same confidence."""
    sets = [frozenset(a.split(", ")) for a in rules["antecedent"]]
    conf = rules["confidence"].to_numpy()
    cons = rules["consequent"].to_numpy()
    keep = []
    for i, a in enumerate(sets):
        redundant = any(
            cons[j] == cons[i] and sets[j] < a and conf[j] >= conf[i] - 1e-9
            for j in range(len(sets))
        )
        keep.append(not redundant)
    return rules[keep]


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    transactions = build_transactions(df)
    comparison = compare_algorithms(transactions)
    rules = mine_rules(transactions)
    good = rating_rules(mine_rules(transactions, min_support=0.005, min_confidence=0.5, min_lift=1.5), "Good")
    poor = rating_rules(mine_rules(transactions, min_support=0.005, min_confidence=0.5, min_lift=1.3), "Poor")
    frequent = fpgrowth(one_hot(transactions), min_support=0.05, use_colnames=True, max_len=3)
    frequent["itemset"] = frequent["itemsets"].map(_fmt)
    frequent["length"] = frequent["itemsets"].map(len)
    frequent = frequent.sort_values("support", ascending=False)[["itemset", "length", "support"]]

    out = {
        "pattern_algorithm_comparison": comparison,
        "frequent_itemsets": frequent.reset_index(drop=True),
        "association_rules": rules,
        "rules_good_rating": good,
        "rules_poor_rating": poor,
    }
    for name, table in out.items():
        table.to_csv(config.TABLE_DIR / f"{name}.csv", index=False)
    return out
