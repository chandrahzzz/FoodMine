"""Generates FoodieMine.ipynb (a step-by-step walkthrough that calls the foodiemine package).

    python notebooks/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/FoodieMine.ipynb
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# FoodieMine: Restaurant Success Analysis using Data Mining\n"
       "BCSE208L Data Mining · Zomato Bangalore dataset\n\n"
       "This notebook walks through every step of the project. The logic lives in the `foodiemine` "
       "package; each cell calls one piece and shows its output."),
    code("import sys, warnings\nsys.path.insert(0, '..')\nwarnings.filterwarnings('ignore')\n"
         "import pandas as pd\nfrom IPython.display import Image, display\n"
         "pd.set_option('display.max_columns', 20); pd.set_option('display.width', 160)\n"
         "pd.set_option('display.max_colwidth', 80)\n"
         "from foodiemine import config, preprocess, warehouse, patterns, classify, cluster, figures\n"
         "config.ensure_dirs()"),

    md("## Step 1: Data preprocessing (Module 3)\n"
       "Clean `rate` (\"4.1/5\", \"NEW\", \"-\") and `approx_cost` (\"1,200\"), merge duplicate listings of "
       "the same restaurant, handle missing values, discretise cost/rating, and create the "
       "*Highly Rated* label (rating ≥ 4.0)."),
    code("raw = preprocess.load_raw()\nprint(raw.shape)\nraw[['name','rate','approx_cost(for two people)','listed_in(type)']].head()"),
    code("raw['rate'].value_counts().head(10)"),
    code("df, unrated, report = preprocess.preprocess(raw)\npreprocess.save(df, unrated, report)\nreport"),
    code("df[['name','location','area','zone','primary_type','service_type','cuisines','rate','rating_level',\n"
         "    'approx_cost','cost_level','votes','popularity','highly_rated']].head()"),

    md("## Step 2: Data warehouse and OLAP (Module 1)\n"
       "Star schema in SQLite: `fact_restaurant` + `dim_location`, `dim_cuisine`, `dim_type`, "
       "`dim_service` (+ a bridge table for the many-to-many restaurant↔cuisine link)."),
    code("warehouse.build(df)"),
    code("warehouse.query('SELECT * FROM dim_location LIMIT 5')"),
    md("**Roll-up**: location → area → zone → city"),
    code("warehouse.rollup('zone')"),
    md("**Drill-down**: South zone → locations"),
    code("warehouse.drill_down('South').head(10)"),
    md("**Slice**: only cafés"),
    code("warehouse.slice_('Cafe')"),
    md("**Dice**: cafés in Koramangala with Low cost"),
    code("warehouse.dice('Cafe', 'Koramangala', 'Low')"),
    md("**Pivot**: average rating by zone × cost level"),
    code("warehouse.pivot('zone', 'cost_level')"),
    code("warehouse.services_effect()"),
    md("**Market gaps**: high demand, low supply"),
    code("olap = warehouse.run_olap_demo()\nolap['market_gaps'].head(10)"),

    md("## Step 3: Frequent pattern mining (Module 4)\n"
       "Each restaurant is a transaction, e.g. `{Cuisine=North Indian, Cost=Medium, OnlineOrder=Yes, Rating=Good}`."),
    code("tx = patterns.build_transactions(df)\ntx[0]"),
    md("Apriori (mlxtend), FP-Growth (mlxtend) and Eclat (own implementation on the vertical tidset format)."),
    code("pats = patterns.run(df)\npats['pattern_algorithm_comparison']"),
    code("pats['frequent_itemsets'].head(15)"),
    md("Rules that lead to a **Good** rating (redundant supersets pruned):"),
    code("pats['rules_good_rating'][['rule','support','confidence','lift']].head(15)"),
    md("Rules that lead to a **Poor** rating:"),
    code("pats['rules_poor_rating'][['rule','support','confidence','lift']].head(10)"),

    md("## Step 4: Classification (Module 5)\n"
       "Predict *Highly Rated* from pre-review features only. Naive Bayes, KNN (lazy learner), "
       "class balancing, bagging ensembles and a Bayesian Belief Network. Evaluated with a stratified "
       "80/20 split plus 5-fold cross-validation. *(about 1 minute)*"),
    code("clf = classify.run(df, unrated)\nclf['metrics']"),
    code("print('Best model by CV F1:', clf['best_name'])\nclf['knn_grid'].sort_values('cv_f1', ascending=False).head()"),
    md("Naive Bayes likelihood ratios: which feature values raise or lower the odds of a high rating."),
    code("clf['signals'].head(12)"),
    md("Bayesian Belief Network: expert structure vs. structure learned by hill-climbing (BIC)."),
    code("print('expert :', clf['bbn']['expert_edges'])\nprint('learned:', clf['bbn']['learned_edges'])\nclf['bbn']['queries']"),
    code("clf['bbn']['diagnostic']"),
    md("Scoring the **new / unrated** restaurants:"),
    code("clf['new_predictions'].head(10)"),

    md("## Step 5: Cluster analysis and outliers (Module 6)\n"
       "Features: rating, log(votes), log(cost), standardised."),
    code("clu = cluster.run(df)\nclu['cluster_k_selection']"),
    code("clu['cluster_comparison']"),
    code("clu['cluster_profile_kmeans']"),
    code("clu['kmedoids_medoids']"),
    code("clu['cluster_profile_dbscan']"),
    code("print(clu['outliers']['outlier_reasons'].value_counts())\n"
         "clu['outliers'][['name','location','rate','votes','approx_cost','outlier_reasons']].head(15)"),

    md("## Figures"),
    code("figures.make_all(df, olap, pats, clf, clu)\n"
         "for f in sorted(config.FIG_DIR.glob('*.png')):\n    print(f.name); display(Image(filename=str(f), width=820))"),

    md("## Step 6: Dashboard\nRun `streamlit run app.py` from the project folder."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).with_name("FoodieMine.ipynb")
nbf.write(nb, out)
print("wrote", out)
