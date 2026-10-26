# FoodieMine: Restaurant Success Analysis using Data Mining

BCSE208L Data Mining project by Chandrahas Reddy (23BDS0308), Slot D2.

FoodieMine studies what makes some Bangalore restaurants highly rated while others are not. It applies
data warehousing, frequent pattern mining, classification and clustering to the Zomato Bangalore
Restaurants dataset from Kaggle (51,717 listings), and presents the results in an interactive dashboard.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python download_data.py
python run_pipeline.py
streamlit run app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

- `download_data.py` downloads zomato.csv (about 550 MB) from Kaggle into data/raw.
- `run_pipeline.py` runs every analysis step and takes about a minute and a half.
- `streamlit run app.py` opens the dashboard in the browser.

A step by step walkthrough with saved outputs is in `notebooks/FoodieMine.ipynb`.

## Project structure

```
foodiemine/
  config.py        paths, thresholds and chart colours
  mappings.py      location to area to zone, and cuisine to cuisine group
  preprocess.py    Step 1: cleaning, de-duplication, discretisation, class label
  warehouse.py     Step 2: SQLite star schema and OLAP operations
  patterns.py      Step 3: Apriori, FP-Growth, Eclat and association rules
  classify.py      Step 4: Naive Bayes, KNN, bagging, Bayesian Belief Network
  cluster.py       Step 5: K-Means, K-Medoids (CLARA), DBSCAN and outliers
  figures.py       report figures
run_pipeline.py    runs all steps
app.py             Step 6: Streamlit dashboard
download_data.py   dataset download
notebooks/         walkthrough notebook
outputs/tables/    results as CSV and JSON
outputs/figures/   report figures
```

## Methods

### Step 1: Data preprocessing

- Kept the columns needed for mining and dropped URL, phone, reviews, menu and dish columns.
- Cleaned the rating column (values such as 4.1/5, NEW and -) and the cost column (values such as 1,200).
- Removed 16,664 exact duplicate rows. The same restaurant is listed once per category such as Delivery
  or Dine-out, so rows were merged on name and address, giving 12,499 unique restaurants.
- Dropped rows with no location or cuisine. Filled missing cost with the median cost of the same
  restaurant type.
- Set aside 2,978 restaurants with no rating as new restaurants, to be scored by the classifier.
- Discretised cost into Low (up to 400), Medium (up to 800) and High; rating into Poor (below 3.5),
  Average and Good (4.0 and above); and votes into Unnoticed, Known and Popular.
- Labelled a restaurant Highly Rated when its rating is 4.0 or above. This covers 22.1 percent of the
  9,503 rated restaurants.

### Step 2: Data warehouse and OLAP

A star schema in SQLite with one fact table (rating, votes, cost) and four dimension tables: location,
cuisine, restaurant type and services. A bridge table links restaurants to their multiple cuisines.
The location hierarchy is location, area, zone and city.

Supported OLAP operations are roll-up, drill-down, slice, dice and pivot. A market gap score finds
areas where a cuisine group gets more customer engagement than usual but is offered by fewer
restaurants than the city norm.

### Step 3: Frequent pattern mining

Each restaurant is treated as a transaction of items such as its cuisines, cost level, online ordering,
table booking, restaurant type, zone and rating level. Apriori and FP-Growth come from mlxtend, and
Eclat is implemented in this project using the vertical data format. All three methods find the same
itemsets at every minimum support tested, and Eclat is the fastest at low support. Rules leading to a
Good or Poor rating are kept, and redundant rules are pruned.

### Step 4: Classification

The models only use information known before a restaurant receives reviews: services, location, type,
listing category, cost and cuisines. This lets the model score new restaurants as well.

| Model | CV accuracy | Test accuracy | Test F1 | ROC AUC |
|---|---|---|---|---|
| Majority baseline | 0.779 | 0.779 | 0.000 | 0.500 |
| Naive Bayes | 0.781 | 0.789 | 0.526 | 0.782 |
| Naive Bayes, balanced priors | 0.745 | 0.764 | 0.547 | 0.782 |
| KNN, k = 9 | 0.825 | 0.816 | 0.511 | 0.754 |
| KNN with oversampling | 0.730 | 0.736 | 0.515 | 0.738 |
| Bagging, KNN | 0.826 | 0.820 | 0.519 | 0.766 |
| Bagging, decision trees | 0.835 | 0.830 | 0.535 | 0.800 |
| Bayesian Belief Network, expert | - | 0.821 | 0.436 | 0.734 |
| Bayesian Belief Network, hill climb | - | 0.820 | 0.431 | 0.717 |

Only 22 percent of restaurants are highly rated, so always predicting Not Highly Rated already gives
77.9 percent accuracy. F1 and ROC AUC are therefore the fairer measures. The dashboard uses the model
with the best cross-validated F1, which is bagging with decision trees.

### Step 5: Clustering and outlier analysis

Restaurants are clustered on rating, votes and cost.

- K-Means with k = 4, chosen by silhouette score. The groups are: Premium, highly rated, popular;
  Budget, average, known; Budget, poorly rated, unnoticed; and Mid-range, poorly rated, unnoticed.
- K-Medoids using CLARA, implemented in this project. It finds the same broad groups and represents
  each group by a real restaurant.
- DBSCAN, with eps chosen from the k-distance curve. Its noise points, together with expensive but
  poorly rated places and cheap places with extreme votes, give 356 outliers.

### Step 6: Dashboard

- Overview: key figures, rating distribution and the effect of services.
- Area explorer: top cuisines and restaurants for an area, plus roll-up, drill-down, slice, dice and pivot.
- Market gaps: under-supplied cuisine groups in each area.
- Association rules: filter rules by support, confidence, lift or item.
- Predict a rating: enter restaurant details to get the chance of a 4.0 or higher rating.
- Clusters and outliers: explore the groups and unusual restaurants.
- Compare my restaurant: compare a restaurant with highly rated peers and get suggestions.
- Model report: all metrics and figures.

## Key findings

- Table booking is the strongest single marker of a high rating. About 71 to 75 percent of restaurants
  that take bookings are highly rated, compared with 13 to 18 percent of those that do not. Booking is
  linked to premium dine-in restaurants, so it marks the segment rather than causing the rating.
- Among restaurants without table booking, those with online ordering are highly rated more often
  (18.3 percent against 13.4 percent).
- The Central zone has the highest share of highly rated restaurants (33.8 percent) and the West zone
  the lowest (13.3 percent).
- Ratings rise with cost in every zone.
- The strongest rules for a Good rating combine table booking with Mediterranean, European, Italian,
  BBQ or Asian cuisine.

## Tools

Python, Pandas, NumPy, SQLite, mlxtend, scikit-learn, pgmpy, Matplotlib, Altair, Streamlit and Jupyter.

## Limitations

The data covers one city at one point in time, around 2019. App ratings can be biased or affected by
fake reviews. The results show patterns and correlations, not causes. The zone mapping is approximate,
and small groups of restaurants give less reliable results.

## References

1. Jiawei Han and Micheline Kamber, Data Mining: Concepts and Techniques, 3rd Edition, Morgan Kaufmann, 2013.
2. Pang-Ning Tan, Michael Steinbach, Anuj Karpatne and Vipin Kumar, Introduction to Data Mining, 2nd Edition, Pearson, 2019.
3. Parteek Bhatia, Data Mining and Data Warehousing: Principles and Practical Techniques, Cambridge University Press, 2019.
4. Zomato Bangalore Restaurants dataset, Kaggle.
