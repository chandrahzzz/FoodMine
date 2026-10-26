"""Static report figures (PNG) for every step, saved to outputs/figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config
from .config import AQUA, BLUE, CATEGORICAL, GRAY, ORANGE, SEQ_BLUE

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, name):
    fig.savefig(config.FIG_DIR / f"{name}.png")
    plt.close(fig)


def rating_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(1.75, 5.06, 0.1)
    ax.hist(df.loc[df.highly_rated == 0, "rate"], bins=bins, color=ORANGE, label="Not highly rated",
            edgecolor="white", linewidth=1)
    ax.hist(df.loc[df.highly_rated == 1, "rate"], bins=bins, color=BLUE, label="Highly rated (>= 4.0)",
            edgecolor="white", linewidth=1)
    ax.axvline(config.HIGH_RATING_THRESHOLD, color=INK_2, linestyle="--", linewidth=1)
    ax.set(title="Rating distribution of 9,503 Bangalore restaurants", xlabel="Rating", ylabel="Restaurants")
    ax.legend()
    _save(fig, "01_rating_distribution")


def heatmap(table: pd.DataFrame, title: str, name: str, fmt="{:.2f}", cbar_label="Average rating"):
    fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 3, 0.45 * table.shape[0] + 1.6))
    vals = table.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap=SEQ_CMAP, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    mid = np.nanmean(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if vals[i, j] > mid else INK)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    ax.set_title(title)
    _save(fig, name)


def services_effect(table: pd.DataFrame):
    t = table.copy()
    t["label"] = "Online: " + t["online_order"] + " / Booking: " + t["book_table"]
    t = t.sort_values("pct_highly_rated")
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(t["label"], t["pct_highly_rated"], color=BLUE, height=0.6)
    for b, n in zip(bars, t["restaurants"]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.0f}%  (n={n:,})", va="center", color=INK_2, fontsize=9)
    ax.set(title="Share of highly rated restaurants by services offered", xlabel="% highly rated", xlim=(0, 100))
    ax.grid(axis="y", visible=False)
    _save(fig, "03_services_effect")


def zone_rollup(table: pd.DataFrame):
    t = table.sort_values("pct_highly_rated")
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(t["zone"], t["pct_highly_rated"], color=BLUE, height=0.6)
    for b, r, n in zip(bars, t["avg_rating"], t["restaurants"]):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.0f}%  avg {r:.2f}  (n={n:,})", va="center", color=INK_2, fontsize=9)
    ax.set(title="Roll-up to zone: % highly rated", xlabel="% highly rated", xlim=(0, 50))
    ax.grid(axis="y", visible=False)
    _save(fig, "02b_zone_rollup")


def algorithm_comparison(table: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.4})
    for algo, color in zip(["Apriori", "FP-Growth", "Eclat"], [BLUE, ORANGE, AQUA]):
        t = table[table.algorithm == algo].sort_values("min_support")
        axes[0].plot(t["min_support"], t["seconds"], marker="o", markersize=6, color=color, label=algo)
        axes[0].annotate(algo, (t["min_support"].iloc[0], t["seconds"].iloc[0]), xytext=(6, 0),
                         textcoords="offset points", color=INK_2, fontsize=9, va="center")
    axes[0].set(title="Run time vs minimum support", xlabel="Minimum support", ylabel="Seconds")
    axes[0].invert_xaxis()
    axes[0].legend()
    t = table[table.algorithm == "Apriori"].sort_values("min_support")
    axes[1].bar([f"{s:.2f}" for s in t["min_support"]], t["itemsets"], color=BLUE, width=0.6)
    for x, v in enumerate(t["itemsets"]):
        axes[1].text(x, v, f"{v:,}", ha="center", va="bottom", color=INK_2, fontsize=9)
    axes[1].set(title="Frequent itemsets found (same for all three)", xlabel="Minimum support",
                ylabel="Itemsets")
    axes[1].invert_xaxis()
    axes[1].grid(axis="x", visible=False)
    _save(fig, "04_pattern_algorithms")


def top_rules(rules: pd.DataFrame, name: str, title: str, n: int = 12):
    t = rules.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.2))
    labels = t["antecedent"].str.replace(", ", "  +  ")
    bars = ax.barh(labels, t["lift"], color=BLUE if "good" in name else ORANGE, height=0.6)
    for b, c, s in zip(bars, t["confidence"], t["support"]):
        ax.text(b.get_width() + 0.03, b.get_y() + b.get_height() / 2,
                f"conf {c:.0%}  supp {s:.1%}", va="center", color=INK_2, fontsize=8)
    ax.set(title=title, xlabel="Lift", xlim=(0, t["lift"].max() * 1.35))
    ax.grid(axis="y", visible=False)
    _save(fig, name)


def model_comparison(metrics: pd.DataFrame):
    t = metrics.dropna(subset=["test_f1"]).sort_values("test_f1")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 * len(t) + 1.5), sharey=True)
    for ax, col, title in ((axes[0], "test_accuracy", "Test accuracy"), (axes[1], "test_f1", "Test F1 (highly rated class)")):
        bars = ax.barh(t["model"], t[col], color=[GRAY if m == "Majority baseline" else BLUE for m in t["model"]],
                       height=0.6)
        for b in bars:
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{b.get_width():.3f}",
                    va="center", color=INK_2, fontsize=8)
        ax.set(title=title, xlim=(0, 1.08))
        ax.grid(axis="y", visible=False)
    _save(fig, "06_model_comparison")


def roc_curves(rocs: dict, metrics: pd.DataFrame, best: str):
    # Three series max: best model, Naive Bayes, KNN - keeps the colours distinguishable.
    knn_name = next(m for m in rocs if m.startswith("KNN (") and "oversampling" not in m)
    chosen = [best, "Naive Bayes", knn_name]
    auc = metrics.set_index("model")["test_roc_auc"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, color in zip(dict.fromkeys(chosen), CATEGORICAL):
        fpr, tpr = rocs[name]
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {auc[name]:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Chance")
    ax.set(title="ROC curves on the test set", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "07_roc_curves")


def confusion(cm, model_name: str):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap=SEQ_CMAP)
    ax.grid(False)
    labels = ["Not highly rated", "Highly rated"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels, rotation=90, va="center")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set(title=f"Confusion matrix\n{model_name}", xlabel="Predicted", ylabel="Actual")
    _save(fig, "08_confusion_matrix")


def nb_signals(signals: pd.DataFrame, n: int = 10):
    t = pd.concat([signals.head(n), signals.tail(n)]).drop_duplicates().sort_values("likelihood_ratio")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.2))
    vals = np.log2(t["likelihood_ratio"])
    ax.barh(t["feature"], vals, color=[BLUE if v > 0 else ORANGE for v in vals], height=0.65)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set(title="Naive Bayes: which features raise or lower the odds of a high rating",
           xlabel="log2 likelihood ratio  P(feature | high) / P(feature | not high)")
    ax.grid(axis="y", visible=False)
    _save(fig, "09_naive_bayes_signals")


def bbn_graph(edges, name: str, title: str):
    import networkx as nx
    g = nx.DiGraph(edges)
    pos = {"Type": (0, 1), "Cost": (1, 2), "OnlineOrder": (1, 0), "TableBooking": (2, 2),
           "Rating": (3, 1), "Popularity": (4, 1)}
    pos = {n: pos.get(n, (2, -1)) for n in g.nodes}
    pos = {n: (x * 1.6, y) for n, (x, y) in pos.items()}
    fig, ax = plt.subplots(figsize=(9, 4))
    size = 3600
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=INK_2, arrows=True, arrowsize=16, width=1.4,
                           node_size=size, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=[BLUE if n == "Rating" else "#cde2fb" for n in g.nodes],
                           node_size=size, edgecolors=BLUE, linewidths=1.2)
    others = {n: n.replace("Booking", "\nBooking").replace("Order", "\nOrder")
              for n in g.nodes if n != "Rating"}
    nx.draw_networkx_labels(g, pos, labels=others, ax=ax, font_size=8, font_color=INK)
    nx.draw_networkx_labels(g, pos, labels={"Rating": "Rating"}, ax=ax, font_size=8, font_color="white")
    ax.set_title(title)
    ax.margins(0.12, 0.22)
    ax.axis("off")
    _save(fig, name)


def k_selection(table: pd.DataFrame, k: int):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(table["k"], table["sse"], marker="o", markersize=6, color=BLUE)
    axes[0].set(title="Elbow method", xlabel="k", ylabel="SSE (within-cluster)")
    axes[1].plot(table["k"], table["silhouette"], marker="o", markersize=6, color=BLUE)
    axes[1].axvline(k, color=INK_2, linestyle="--", linewidth=1)
    axes[1].annotate(f"chosen k = {k}", (k, table["silhouette"].max()), xytext=(6, 0),
                     textcoords="offset points", color=INK_2, fontsize=9)
    axes[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
    _save(fig, "10_k_selection")


def cluster_facets(assign: pd.DataFrame, names: dict):
    """Small multiples: each cluster highlighted against all restaurants in gray."""
    ks = sorted(names)
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 3.8), sharex=True, sharey=True)
    for ax, c in zip(np.atleast_1d(axes), ks):
        ax.scatter(assign["approx_cost"], assign["rate"], s=4, color="#d6d5d0", linewidths=0)
        sel = assign[assign["kmeans_cluster"] == c]
        ax.scatter(sel["approx_cost"], sel["rate"], s=6, color=BLUE, alpha=0.6, linewidths=0)
        ax.set_xscale("log")
        ax.set_title(f"{names[c]}\n{len(sel):,} restaurants", fontsize=10)
        ax.set_xlabel("Cost for two (log scale)")
    np.atleast_1d(axes)[0].set_ylabel("Rating")
    fig.suptitle("K-Means clusters (cost vs rating)", fontweight="bold", y=1.04)
    _save(fig, "11_kmeans_clusters")


def dbscan_figs(kdist: np.ndarray, eps: float, assign: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(np.arange(len(kdist)), kdist, color=BLUE)
    axes[0].axhline(eps, color=INK_2, linestyle="--", linewidth=1)
    axes[0].annotate(f"eps = {eps:.3f}", (0, eps), xytext=(4, 4), textcoords="offset points",
                     color=INK_2, fontsize=9)
    axes[0].set(title="Sorted k-distance (knee picks eps)", xlabel="Points sorted by distance",
                ylabel="Distance to 6th neighbour")
    noise = assign["dbscan_cluster"] == -1
    axes[1].scatter(assign.loc[~noise, "votes"] + 1, assign.loc[~noise, "rate"], s=4, color="#b7d3f6",
                    linewidths=0, label="In a dense cluster")
    axes[1].scatter(assign.loc[noise, "votes"] + 1, assign.loc[noise, "rate"], s=12, color=ORANGE,
                    linewidths=0, label=f"Noise / outlier ({noise.sum()})")
    axes[1].set_xscale("log")
    axes[1].set(title="DBSCAN outliers", xlabel="Votes + 1 (log scale)", ylabel="Rating")
    axes[1].legend(markerscale=2)
    _save(fig, "12_dbscan_outliers")


def make_all(df, olap, pats, clf, clu):
    config.ensure_dirs()
    rating_distribution(df)
    heatmap(olap["olap_pivot_zone_cost"].set_index("zone"), "Pivot: average rating by zone and cost level",
            "02_pivot_zone_cost")
    zone_rollup(olap["olap_rollup_zone"])
    services_effect(olap["olap_services_effect"])
    algorithm_comparison(pats["pattern_algorithm_comparison"])
    top_rules(pats["rules_good_rating"], "05_rules_good_rating", "Strongest rules leading to a Good rating (>= 4.0)")
    top_rules(pats["rules_poor_rating"], "05b_rules_poor_rating", "Strongest rules leading to a Poor rating (< 3.5)")
    model_comparison(clf["metrics"])
    roc_curves(clf["rocs"], clf["metrics"], clf["best_name"])
    confusion(clf["confusions"][clf["best_name"]], clf["best_name"])
    nb_signals(clf["signals"])
    bbn_graph(clf["bbn"]["expert_edges"], "09b_bbn_expert", "Bayesian Belief Network (expert structure)")
    k_selection(clu["cluster_k_selection"], clu["k"])
    cluster_facets(clu["cluster_assignments"], clu["kmeans_names"])
    dbscan_figs(clu["kdist"], clu["eps"], clu["cluster_assignments"])
