"""Run the whole FoodieMine pipeline end to end.

    python run_pipeline.py              # all steps
    python run_pipeline.py --skip-raw   # reuse data/processed instead of re-cleaning the raw CSV

Outputs: data/processed/*.csv, data/warehouse/foodiemine.db,
         outputs/tables/*, outputs/figures/*.png, outputs/models/*.joblib
"""
import argparse
import time
import warnings

import pandas as pd

from foodiemine import classify, cluster, config, figures, patterns, preprocess, warehouse

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_colwidth", 70)


def step(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    return time.perf_counter()


def main(skip_raw: bool) -> None:
    config.ensure_dirs()
    t0 = time.perf_counter()

    t = step("Step 1  Preprocessing")
    if skip_raw and config.CLEAN_CSV.exists():
        print("Using existing", config.CLEAN_CSV)
    else:
        if not config.RAW_CSV.exists():
            raise SystemExit(f"Raw data missing: {config.RAW_CSV}\nRun: python download_data.py")
        _, _, report = preprocess.run()
        for k in ("raw_rows", "exact_duplicate_rows", "unique_restaurants", "unrated_new_restaurants",
                  "final_rated_restaurants", "highly_rated_share"):
            print(f"  {k:28s} {report[k]}")
    df = preprocess.load_clean()
    unrated = preprocess.load_clean(config.UNRATED_CSV)
    print(f"  done in {time.perf_counter() - t:.1f}s")

    t = step("Step 2  Star schema warehouse + OLAP")
    print("  tables:", warehouse.build(df))
    olap = warehouse.run_olap_demo()
    print(olap["olap_rollup_zone"].to_string(index=False))
    print(f"  done in {time.perf_counter() - t:.1f}s")

    t = step("Step 3  Frequent patterns (Apriori / FP-Growth / Eclat)")
    pats = patterns.run(df)
    print(pats["pattern_algorithm_comparison"].to_string(index=False))
    print("\n  Top rules -> Rating=Good")
    print(pats["rules_good_rating"][["rule", "support", "confidence", "lift"]].head(8).round(3).to_string(index=False))
    print(f"  done in {time.perf_counter() - t:.1f}s")

    t = step("Step 4  Classification")
    clf = classify.run(df, unrated)
    print(clf["metrics"][["model", "cv_accuracy", "test_accuracy", "test_precision", "test_recall",
                          "test_f1", "test_roc_auc"]].to_string(index=False))
    print("  best model (by CV F1):", clf["best_name"])
    print(f"  done in {time.perf_counter() - t:.1f}s")

    t = step("Step 5  Clustering + outliers")
    clu = cluster.run(df)
    print(clu["cluster_comparison"].to_string(index=False))
    print(clu["cluster_profile_kmeans"][["name", "restaurants", "median_rating", "median_votes",
                                         "median_cost"]].to_string(index=False))
    print("  outliers:", len(clu["outliers"]))
    print(f"  done in {time.perf_counter() - t:.1f}s")

    t = step("Figures")
    figures.make_all(df, olap, pats, clf, clu)
    print("  saved to", config.FIG_DIR)

    print(f"\nPipeline finished in {time.perf_counter() - t0:.0f}s. "
          "Launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-raw", action="store_true", help="reuse data/processed if it exists")
    main(ap.parse_args().skip_raw)
