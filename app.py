"""FoodieMine dashboard.   Run:  streamlit run app.py   (after python run_pipeline.py)"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from foodiemine import classify, config, preprocess, warehouse
from foodiemine.config import BLUE, GRAY, ORANGE

st.set_page_config(page_title="FoodieMine", page_icon="🍽️", layout="wide")

T = config.TABLE_DIR
REQUIRED = [config.CLEAN_CSV, config.WAREHOUSE_DB, config.MODEL_DIR / "rating_model.joblib",
            T / "cluster_assignments.csv"]
if not all(p.exists() for p in REQUIRED):
    st.error("Pipeline outputs not found. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()


# ---------------------------------------------------------------- loaders -----
@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(T / f"{name}.csv")


@st.cache_data
def details(name: str) -> dict:
    with open(T / f"{name}.json") as f:
        return json.load(f)


@st.cache_data
def restaurants() -> pd.DataFrame:
    return preprocess.load_clean()


@st.cache_data
def sql(q: str, params: tuple = ()) -> pd.DataFrame:
    return warehouse.query(q, params)


@st.cache_resource
def model_bundle() -> dict:
    return classify.load_model()


def bar(df, x, y, title="", color=BLUE, fmt=".2f", sort="-x", height=None, tooltip=None):
    base = alt.Chart(df).encode(
        x=alt.X(x, title=None),
        y=alt.Y(y, sort=sort, title=None),
        tooltip=tooltip or list(df.columns),
    )
    chart = base.mark_bar(color=color, cornerRadiusEnd=3, height={"band": 0.7}) + base.mark_text(
        align="left", dx=4, color="gray").encode(text=alt.Text(x.split(":")[0], format=fmt))
    return chart.properties(title=title, height=height or max(140, 26 * len(df)))


def kpis(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


df = restaurants()

st.sidebar.title("🍽️ FoodieMine")
st.sidebar.caption("Restaurant success analysis · Zomato Bangalore")
page = st.sidebar.radio("Go to", [
    "Overview", "Area explorer (OLAP)", "Market gaps", "Association rules",
    "Predict a rating", "Clusters & outliers", "Compare my restaurant", "Model report",
])
st.sidebar.divider()
st.sidebar.caption(f"{len(df):,} rated restaurants · {df['location'].nunique()} locations · "
                   f"rating ≥ {config.HIGH_RATING_THRESHOLD} = highly rated")


# ================================================================= Overview ===
if page == "Overview":
    st.title("What makes a Bangalore restaurant successful?")
    rep = details("preprocessing_report")
    kpis([
        ("Raw listings", f"{rep['raw_rows']:,}"),
        ("Unique restaurants", f"{rep['unique_restaurants']:,}"),
        ("Rated (analysed)", f"{rep['final_rated_restaurants']:,}"),
        ("Highly rated", f"{rep['highly_rated_share']:.1%}"),
        ("Average rating", f"{df['rate'].mean():.2f}"),
    ])
    st.caption(f"{rep['unrated_new_restaurants']:,} restaurants are new or unrated. They are kept aside "
               "and scored by the prediction model (see *Predict a rating*).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rating distribution")
        hist = df.assign(group=np.where(df.highly_rated == 1, "Highly rated", "Not highly rated"))
        st.altair_chart(
            alt.Chart(hist).mark_bar(binSpacing=1).encode(
                x=alt.X("rate:Q", bin=alt.Bin(step=0.1), title="Rating"),
                y=alt.Y("count():Q", title="Restaurants"),
                color=alt.Color("group:N", scale=alt.Scale(domain=["Highly rated", "Not highly rated"],
                                                           range=[BLUE, ORANGE]), title=None,
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("count():Q", title="Restaurants"), "group:N"],
            ).properties(height=280), width="stretch")
    with c2:
        st.subheader("Do services matter?")
        s = table("olap_services_effect")
        s["services"] = "Online " + s.online_order + " · Booking " + s.book_table
        st.altair_chart(bar(s, "pct_highly_rated:Q", "services:N", fmt=".0f",
                            tooltip=["services", "restaurants", "avg_rating", "avg_cost", "pct_highly_rated"]),
                        width="stretch")
        st.caption("Table booking is the strongest single signal, but it travels with higher prices "
                   "and dine-in formats, so it is a marker of the segment rather than a cause.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By zone (roll-up)")
        z = table("olap_rollup_zone")
        st.altair_chart(bar(z, "pct_highly_rated:Q", "zone:N", fmt=".1f"), width="stretch")
    with c2:
        st.subheader("Average rating: restaurant type × cost")
        p = table("olap_pivot_type_cost").set_index("rest_type")
        counts = df["primary_type"].value_counts()
        p = p.loc[[t for t in counts.index[:12] if t in p.index]]
        st.dataframe(p.style.background_gradient(cmap="Blues", axis=None).format("{:.2f}", na_rep="–"),
                     width="stretch")


# ====================================================== Area explorer (OLAP) ==
elif page == "Area explorer (OLAP)":
    st.title("Area explorer")
    areas = df["area"].value_counts()
    area = st.selectbox("Choose an area", areas.index, index=list(areas.index).index("Koramangala")
                        if "Koramangala" in areas.index else 0)
    a = df[df["area"] == area]
    kpis([
        ("Restaurants", f"{len(a):,}"),
        ("Average rating", f"{a['rate'].mean():.2f}"),
        ("Highly rated", f"{a['highly_rated'].mean():.1%}"),
        ("Median cost for two", f"₹{a['approx_cost'].median():,.0f}"),
        ("Online ordering", f"{a['online_order'].mean():.0%}"),
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Top cuisines in {area}")
        cz = warehouse.cuisine_by_area(area, min_restaurants=5).head(12)
        st.altair_chart(bar(cz, "restaurants:Q", "cuisine:N", fmt=",.0f"), width="stretch")
    with c2:
        st.subheader("Best-rated cuisines (≥ 5 restaurants)")
        st.altair_chart(bar(cz.sort_values("avg_rating", ascending=False), "avg_rating:Q", "cuisine:N"),
                        width="stretch")
    st.subheader(f"Top restaurants in {area}")
    st.dataframe(a.sort_values(["rate", "votes"], ascending=False)[
        ["name", "location", "primary_type", "cuisines", "rate", "votes", "approx_cost"]].head(15),
        hide_index=True, width="stretch")

    st.divider()
    st.header("OLAP operations on the star schema")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Roll-up", "Drill-down", "Slice", "Dice", "Pivot"])
    with tab1:
        level = st.select_slider("Geography level", ["location", "area", "zone", "city"], value="zone")
        st.caption("location → area → zone → city. Moving right rolls up into fewer, bigger groups.")
        st.dataframe(warehouse.rollup(level), hide_index=True, width="stretch")
    with tab2:
        zone = st.selectbox("Zone", ["South", "East", "North", "Central", "West"])
        st.dataframe(warehouse.drill_down(zone), hide_index=True, width="stretch")
    with tab3:
        rt = st.selectbox("Restaurant type", df["primary_type"].value_counts().index, index=2)
        st.dataframe(warehouse.slice_(rt), hide_index=True, width="stretch")
    with tab4:
        c1, c2, c3 = st.columns(3)
        d_type = c1.selectbox("Type", df["primary_type"].value_counts().index, index=2, key="dt")
        d_area = c2.selectbox("Area", areas.index, index=list(areas.index).index("Koramangala"), key="da")
        d_cost = c3.selectbox("Cost level", config.COST_LABELS, key="dc")
        res = warehouse.dice(d_type, d_area, d_cost)
        st.caption(f"{len(res)} restaurants")
        st.dataframe(res, hide_index=True, width="stretch")
    with tab5:
        dims = ["zone", "cost_level", "rest_type", "service_type", "online_order", "book_table"]
        c1, c2, c3 = st.columns(3)
        r = c1.selectbox("Rows", dims, index=0)
        c = c2.selectbox("Columns", [d for d in dims if d != r], index=0)
        m = c3.selectbox("Measure", ["avg_rating", "pct_highly_rated", "restaurants", "avg_votes", "avg_cost"])
        pv = warehouse.pivot(r, c, m)
        st.dataframe(pv.style.background_gradient(cmap="Blues", axis=None).format("{:.2f}", na_rep="–"),
                     width="stretch")


# ============================================================= Market gaps ===
elif page == "Market gaps":
    st.title("Market gaps")
    st.markdown(
        "Cells where a cuisine group draws **more customer engagement than usual** (votes) in an area, "
        "but **fewer restaurants** there offer it than across the city. Both indexes are relative to the "
        "city norm of 1.0.\n\n"
        "`gap score = demand index ÷ supply index`, so higher means a bigger opportunity.")
    g = table("market_gaps")
    st.altair_chart(
        alt.Chart(g.head(15).assign(cell=g.area + " · " + g.cuisine_group)).mark_bar(
            color=BLUE, cornerRadiusEnd=3).encode(
            x=alt.X("gap_score:Q", title="Gap score"),
            y=alt.Y("cell:N", sort="-x", title=None),
            tooltip=list(g.columns)).properties(height=420),
        width="stretch")
    st.dataframe(g, hide_index=True, width="stretch")
    st.caption("Small cells are noisy: treat a high score as a lead to investigate, not a guarantee.")


# ======================================================= Association rules ===
elif page == "Association rules":
    st.title("Association rules")
    st.caption("Each restaurant is a transaction of items: cuisines, cost level, services, type, zone, rating.")
    which = st.radio("Rules", ["Lead to a Good rating", "Lead to a Poor rating", "All rules"], horizontal=True)
    rules = table({"Lead to a Good rating": "rules_good_rating",
                   "Lead to a Poor rating": "rules_poor_rating",
                   "All rules": "association_rules"}[which])
    c1, c2, c3, c4 = st.columns(4)
    min_s = c1.slider("Min support", 0.0, 0.10, 0.005, 0.001, format="%.3f")
    min_c = c2.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)
    min_l = c3.slider("Min lift", 1.0, 5.0, 1.2, 0.1)
    term = c4.text_input("Contains item", placeholder="e.g. Cafe, OnlineOrder=Yes")
    r = rules[(rules.support >= min_s) & (rules.confidence >= min_c) & (rules.lift >= min_l)]
    if term:
        r = r[r["rule"].str.contains(term, case=False, regex=False)]
    st.caption(f"{len(r):,} rules")
    st.dataframe(r[["rule", "support", "confidence", "lift", "leverage", "conviction"]].round(3),
                 hide_index=True, width="stretch", height=420)

    st.subheader("Apriori vs FP-Growth vs Eclat")
    comp = table("pattern_algorithm_comparison")
    st.altair_chart(
        alt.Chart(comp).mark_line(point=alt.OverlayMarkDef(size=60)).encode(
            x=alt.X("min_support:Q", scale=alt.Scale(reverse=True), title="Minimum support"),
            y=alt.Y("seconds:Q", title="Seconds"),
            color=alt.Color("algorithm:N", scale=alt.Scale(domain=["Apriori", "FP-Growth", "Eclat"],
                                                           range=[BLUE, ORANGE, config.AQUA]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=list(comp.columns)).properties(height=280),
        width="stretch")
    st.dataframe(comp.pivot(index="min_support", columns="algorithm", values="seconds")
                 .join(comp.groupby("min_support")[["itemsets", "all_three_agree"]].first())
                 .sort_index(ascending=False), width="stretch")


# ======================================================== Predict a rating ===
elif page == "Predict a rating":
    st.title("Will this restaurant be highly rated?")
    bundle = model_bundle()
    spec = bundle["spec"]
    st.caption(f"Model: **{bundle['name']}**, chosen by cross-validated F1. It only uses information "
               "available before any reviews, so it works for brand-new restaurants.")
    tab_form, tab_new = st.tabs(["Try a restaurant", "Score the unrated restaurants"])
    with tab_form:
        cats = spec["categories"]
        all_cuisines = sorted(df["cuisine_list"].explode().unique())
        with st.form("predict"):
            c1, c2, c3 = st.columns(3)
            location = c1.selectbox("Location", cats["location"], index=cats["location"].index("Koramangala 5th Block"))
            rtype = c2.selectbox("Restaurant type", cats["primary_type"], index=cats["primary_type"].index("Cafe"))
            service = c3.selectbox("Main listing category", cats["service_type"],
                                   index=cats["service_type"].index("Cafes"))
            c1, c2, c3 = st.columns(3)
            cost = c1.number_input("Approx. cost for two (₹)", 50, 6000, 700, step=50)
            online = c2.toggle("Online ordering", True)
            booking = c3.toggle("Table booking", False)
            cuisines = st.multiselect("Cuisines", all_cuisines, default=["Cafe", "Italian", "Continental"])
            go = st.form_submit_button("Predict", type="primary")
        if go:
            if not cuisines:
                st.warning("Pick at least one cuisine.")
                st.stop()
            row = pd.DataFrame([{
                "online_order": int(online), "book_table": int(booking), "location": location,
                "primary_type": rtype, "service_type": service, "approx_cost": float(cost),
                "cuisine_list": cuisines,
            }])
            X = classify.make_features(row, spec)
            p = float(bundle["model"].predict_proba(X)[0, 1])
            p_nb = float(bundle["nb"].predict_proba(X)[0, 1])
            base = df["highly_rated"].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Chance of ≥ 4.0 rating", f"{p:.0%}", f"{(p - base) * 100:+.0f} pts vs city average")
            c2.metric("Naive Bayes estimate", f"{p_nb:.0%}")
            c3.metric("City average", f"{base:.0%}")
            verdict = "likely to be **highly rated**" if p >= 0.5 else "**not** likely to reach 4.0"
            st.markdown(f"This restaurant is {verdict}.")

            peers = df[(df["area"] == df.loc[df["location"] == location, "area"].iloc[0])
                       & (df["primary_type"] == rtype)]
            if len(peers) >= 5:
                hi = peers[peers.highly_rated == 1]
                st.subheader(f"{rtype} restaurants in the same area ({len(peers)})")
                comp = pd.DataFrame({
                    "": ["Online ordering", "Table booking", "Median cost for two"],
                    "Highly rated peers": [f"{hi.online_order.mean():.0%}" if len(hi) else "–",
                                           f"{hi.book_table.mean():.0%}" if len(hi) else "–",
                                           f"₹{hi.approx_cost.median():,.0f}" if len(hi) else "–"],
                    "Other peers": [f"{peers[peers.highly_rated == 0].online_order.mean():.0%}",
                                    f"{peers[peers.highly_rated == 0].book_table.mean():.0%}",
                                    f"₹{peers[peers.highly_rated == 0].approx_cost.median():,.0f}"],
                    "This restaurant": ["Yes" if online else "No", "Yes" if booking else "No", f"₹{cost:,.0f}"],
                })
                st.dataframe(comp, hide_index=True, width="stretch")

        st.subheader("Bayesian Belief Network: P(highly rated | cost, booking, online order)")
        q = table("bbn_queries")
        st.dataframe(q.style.background_gradient(cmap="Blues", subset=["P(Highly rated)"]).format(
            {"P(Highly rated)": "{:.0%}"}), hide_index=True, width="stretch")
        st.caption("Rows backed by only a handful of training restaurants are unreliable.")
    with tab_new:
        new = table("new_restaurant_predictions")
        st.caption(f"{len(new):,} restaurants with no rating yet ('NEW' or '-'). Sorted by predicted chance of "
                   "becoming highly rated.")
        q = st.text_input("Search name or location")
        if q:
            new = new[new["name"].str.contains(q, case=False, regex=False)
                      | new["location"].str.contains(q, case=False, regex=False)]
        st.dataframe(new, hide_index=True, width="stretch", height=500,
                     column_config={"p_highly_rated": st.column_config.ProgressColumn(
                         "P(highly rated)", min_value=0, max_value=1, format="percent")})


# ====================================================== Clusters & outliers ===
elif page == "Clusters & outliers":
    st.title("Restaurant groups and unusual restaurants")
    assign = table("cluster_assignments")
    prof = table("cluster_profile_kmeans")
    st.subheader("K-Means groups")
    st.dataframe(prof, hide_index=True, width="stretch")
    pick = st.selectbox("Highlight a group", prof["name"])
    cid = int(prof.loc[prof.name == pick, "cluster"].iloc[0])
    plot = assign.assign(group=np.where(assign.kmeans_cluster == cid, pick, "Other restaurants"))
    plot = plot.sort_values("group", key=lambda s: s != "Other restaurants")
    st.altair_chart(
        alt.Chart(plot).mark_circle(size=14, opacity=0.7).encode(
            x=alt.X("approx_cost:Q", scale=alt.Scale(type="log"), title="Cost for two (log)"),
            y=alt.Y("rate:Q", scale=alt.Scale(domain=[1.8, 5]), title="Rating"),
            color=alt.Color("group:N", scale=alt.Scale(domain=[pick, "Other restaurants"],
                                                       range=[BLUE, "#d6d5d0"]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["name", "location", "primary_type", "rate", "votes", "approx_cost", "kmeans_name"],
        ).properties(height=420).interactive(), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Method comparison")
        st.dataframe(table("cluster_comparison"), hide_index=True, width="stretch")
        cd = details("cluster_details")
        st.caption(f"Adjusted Rand index K-Means vs K-Medoids: {cd['ARI K-Means vs K-Medoids']:.2f} · "
                   f"DBSCAN eps={cd['dbscan_eps']}, min_samples={cd['dbscan_min_samples']}")
    with c2:
        st.subheader("K-Medoids representatives")
        st.caption("Medoids are real restaurants, so each group has a concrete example.")
        st.dataframe(table("kmedoids_medoids"), hide_index=True, width="stretch")

    st.subheader("Outliers")
    out = table("outliers")
    kinds = ["DBSCAN noise", "Expensive but poorly rated", "Cheap with extreme votes"]
    chosen = st.multiselect("Reason", kinds, default=kinds)
    out = out[out["outlier_reasons"].apply(lambda s: any(k in s for k in chosen))]
    st.caption(f"{len(out)} restaurants. DBSCAN noise includes hidden gems (huge votes, top ratings) "
               "as well as places with odd cost/rating combinations.")
    st.dataframe(out[["name", "location", "primary_type", "rate", "votes", "approx_cost", "outlier_reasons"]],
                 hide_index=True, width="stretch", height=400)


# ==================================================== Compare my restaurant ===
elif page == "Compare my restaurant":
    st.title("Compare a restaurant with its peers")
    assign = table("cluster_assignments")
    names = (df["name"] + " — " + df["location"]).tolist()
    choice = st.selectbox("Restaurant", names, index=None, placeholder="Start typing a name…")
    if choice:
        me = df.iloc[names.index(choice)]
        my_cluster = assign.loc[assign.restaurant_id == me.restaurant_id, "kmeans_name"].iloc[0]
        kpis([("Rating", f"{me.rate:.1f}"), ("Votes", f"{me.votes:,}"),
              ("Cost for two", f"₹{me.approx_cost:,.0f}"), ("Group", my_cluster)])
        peers = df[(df["area"] == me["area"]) & (df["primary_type"] == me["primary_type"])
                   & (df["restaurant_id"] != me["restaurant_id"])]
        hi = peers[peers.highly_rated == 1]
        st.subheader(f"{me.primary_type} restaurants in {me.area}: {len(peers)} peers, {len(hi)} highly rated")
        if len(hi) == 0:
            st.info("No highly rated peers of the same type in this area to compare with.")
        else:
            rows = [
                ("Offers online ordering", "Yes" if me.online_order else "No", f"{hi.online_order.mean():.0%}"),
                ("Offers table booking", "Yes" if me.book_table else "No", f"{hi.book_table.mean():.0%}"),
                ("Cost for two", f"₹{me.approx_cost:,.0f}", f"median ₹{hi.approx_cost.median():,.0f}"),
                ("Number of cuisines", str(me.n_cuisines), f"median {hi.n_cuisines.median():.0f}"),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["", "This restaurant", "Highly rated peers"]),
                         hide_index=True, width="stretch")
            tips = []
            if not me.online_order and hi.online_order.mean() >= 0.6:
                tips.append(f"{hi.online_order.mean():.0%} of highly rated peers offer online ordering; this one does not.")
            if not me.book_table and hi.book_table.mean() >= 0.5:
                tips.append(f"{hi.book_table.mean():.0%} of highly rated peers take table bookings.")
            peer_cuisines = hi["cuisine_list"].explode().value_counts(normalize=True)
            missing = [c for c in peer_cuisines.index[:5] if c not in me.cuisine_list and peer_cuisines[c] >= 0.3]
            if missing:
                tips.append("Common among highly rated peers but not served here: " + ", ".join(missing) + ".")
            if me.highly_rated:
                st.success("Already highly rated.")
            for t in tips:
                st.markdown(f"- {t}")
            if not tips and not me.highly_rated:
                st.markdown("Services and cuisines already match the highly rated peers; the gap is likely "
                            "food, service quality or ambience, which this data does not capture.")
            st.caption("These are patterns, not proof that a change will raise the rating.")
            st.dataframe(hi.sort_values("rate", ascending=False)[
                ["name", "location", "cuisines", "rate", "votes", "approx_cost", "online_order", "book_table"]].head(10),
                hide_index=True, width="stretch")


# ============================================================ Model report ===
elif page == "Model report":
    st.title("Model report")
    m = table("classification_metrics")
    cd = details("classification_details")
    st.markdown(f"Train {cd['train_size']:,} / test {cd['test_size']:,} restaurants (stratified 80/20). "
                f"Only {cd['positive_share']:.0%} are highly rated, so the **majority baseline** already "
                f"scores {m.loc[m.model == 'Majority baseline', 'test_accuracy'].iloc[0]:.1%} accuracy; "
                "compare F1 and ROC-AUC, not accuracy alone. Best model by CV F1: "
                f"**{cd['best_model']}**.")
    st.dataframe(m.style.format(precision=3, na_rep="–").highlight_max(
        subset=["cv_accuracy", "cv_f1", "test_accuracy", "test_f1", "test_roc_auc"], color="#cde2fb"),
        hide_index=True, width="stretch")
    figs = [
        ("06_model_comparison", "Model comparison"), ("07_roc_curves", "ROC curves"),
        ("08_confusion_matrix", "Confusion matrix"), ("09_naive_bayes_signals", "Naive Bayes signals"),
        ("09b_bbn_expert", "Bayesian Belief Network"), ("10_k_selection", "Choosing k"),
        ("11_kmeans_clusters", "K-Means clusters"), ("12_dbscan_outliers", "DBSCAN"),
        ("04_pattern_algorithms", "Pattern-mining algorithms"), ("02_pivot_zone_cost", "OLAP pivot"),
    ]
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, (f, cap) in zip(cols, figs[i:i + 2]):
            path = config.FIG_DIR / f"{f}.png"
            if path.exists():
                col.image(str(path), caption=cap, width="stretch")
