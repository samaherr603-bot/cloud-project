"""
Building Energy Efficiency: Streamlit dashboard + inference app
(Building Data Genome Project 2)

Files this app needs in the same GitHub folder:
  streamlit_app.py      <- this file (the "Main file path" on Streamlit Cloud)
  bdg_core.py           <- preprocessing, inference and charts (from the notebook)
  model_bundle.joblib   <- trained models + preprocessor
  buildings.csv         <- one row per building, for the dashboard and look-ups
  requirements.txt
"""
import numpy as np
import pandas as pd
import streamlit as st

import bdg_core as core

st.set_page_config(page_title="Building Energy Efficiency", page_icon="🏢", layout="wide")

MODELS = core.MODEL_NAMES[:3]   # the three trained models (naive baseline excluded)


# ----------------------------------------------------------------------------
# Load the artifacts once per server (shared by all visitors)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def get_bundle():
    return core.load_bundle()


@st.cache_data(show_spinner="Loading data…")
def get_buildings():
    return core.load_buildings().copy()


try:
    bundle = get_bundle()
    df = get_buildings()
except Exception as e:  # most often: missing file or library version mismatch
    st.error(
        "Could not load `model_bundle.joblib` / `buildings.csv`. Check that both files "
        "are in the repository next to this file, and that `requirements.txt` pins the "
        f"same scikit-learn version used in Colab.\n\nDetails: `{e}`"
    )
    st.stop()

cats = bundle["prep"]["categories"]
best = bundle["metrics"]["Neural network"]


def probs_for(raw):
    """{model name: P(efficient)} for a one-row DataFrame."""
    return {m: float(core.predict_proba(raw, m)[0]) for m in MODELS}


def show_probabilities(probs):
    cols = st.columns(len(probs))
    for col, (name, p) in zip(cols, probs.items()):
        col.metric(name, core.LABELS[int(p > 0.5)], f"P(efficient) = {p:.1%}",
                   delta_color="off")
    st.pyplot(core.fig_prediction(probs))


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🏢 Building energy efficiency")
st.caption(
    "Predicts whether a building uses less energy per square foot than the national "
    "median for its type, using data from the Building Data Genome Project 2."
)

tab_dash, tab_models, tab_new, tab_lookup, tab_batch = st.tabs(
    ["📊 Dashboard", "🧠 Model performance", "➕ Predict a new building",
     "🔎 Look up a building", "📁 Batch prediction"]
)

# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
with tab_dash:
    sites = sorted(df["site"].unique())
    chosen = st.multiselect("Filter by site", sites, default=sites)
    view = df[df["site"].isin(chosen)]
    if view.empty:
        st.info("Pick at least one site.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Buildings", f"{len(view):,}")
        k2.metric("Sites", view["site"].nunique())
        k3.metric("Labeled efficient", f"{view['isEfficient'].mean():.0%}")
        k4.metric("Median source EUI",
                  f"{view.loc[view['source_eui_est'] > 0, 'source_eui_est'].median():.1f}")

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(core.fig_share_by_use(view))
        with c2:
            st.pyplot(core.fig_eui_by_use(view))
        st.pyplot(core.fig_seasonal(view))
        st.pyplot(core.fig_site_share(view))

        with st.expander("Show the data"):
            st.dataframe(view, hide_index=True)
            st.download_button("Download as CSV", view.to_csv(index=False),
                               "buildings_filtered.csv", "text/csv")

# ----------------------------------------------------------------------------
# Model performance
# ----------------------------------------------------------------------------
with tab_models:
    k1, k2, k3 = st.columns(3)
    k1.metric("Best model test accuracy", f"{best['Accuracy']:.1%}")
    k2.metric("Naive baseline accuracy", f"{bundle['metrics']['Naive baseline']['Accuracy']:.1%}")
    k3.metric("Not-efficient buildings caught", f"{best['Recall (not efficient)']:.1%}")

    st.pyplot(core.fig_model_comparison(bundle))
    st.dataframe(pd.DataFrame(bundle["metrics"]).T.round(3))

    c1, c2 = st.columns(2)
    with c1:
        cm_model = st.selectbox("Confusion matrix for", core.MODEL_NAMES)
        st.pyplot(core.fig_confusion(cm_model))
    with c2:
        st.pyplot(core.fig_logistic_weights())

# ----------------------------------------------------------------------------
# Predict a new building
# ----------------------------------------------------------------------------
with tab_new:
    st.write("Enter what you know about the building. Leave a number blank if it is "
             "unknown; blanks are treated the same way as missing values in training.")
    med = df[core.NUMERIC_COLS + core.METER_COLS].median(numeric_only=True)

    def default_index(options, preferred):
        return options.index(preferred) if preferred in options else 0

    with st.form("new_building"):
        values = {}
        c1, c2, c3 = st.columns(3)
        values["primaryspaceusage"] = c1.selectbox(
            "Primary use", cats["primaryspaceusage"],
            index=default_index(cats["primaryspaceusage"], "Education"))
        values["sub_primaryspaceusage"] = c2.selectbox(
            "Sub-use", cats["sub_primaryspaceusage"],
            index=default_index(cats["sub_primaryspaceusage"], "Academic"))
        values["timezone"] = c3.selectbox(
            "Timezone", cats["timezone"],
            index=default_index(cats["timezone"], "US/Eastern"))

        cols = st.columns(len(core.NUMERIC_COLS))
        for col, c in zip(cols, core.NUMERIC_COLS):
            v = med.get(c)
            values[c] = col.number_input(c, value=None if pd.isna(v) else float(round(v, 2)))

        st.markdown("**Average daily meter readings, kWh** (first week of Jan / Apr / Jul / Oct)")
        for meter in core.METERS:
            cols = st.columns(len(core.MONTHS))
            for col, mo in zip(cols, core.MONTHS):
                c = f"{meter}_{mo}"
                v = med.get(c)
                values[c] = col.number_input(
                    c, min_value=0.0, value=0.0 if pd.isna(v) else float(round(v, 2)))

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw = pd.DataFrame([values])[core.INPUT_COLS]
        show_probabilities(probs_for(raw))
        meters = raw[core.METER_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
        if (meters.to_numpy() == 0).all():
            st.info("No meter readings were entered, so the **Neural network (no meter "
                    "data)** is the most reliable model for this building.")
        for msg in core.unknown_categories(raw):
            st.warning(f"Unseen category, ignored by the models: {msg}")

# ----------------------------------------------------------------------------
# Look up a building from the dataset
# ----------------------------------------------------------------------------
with tab_lookup:
    bid = st.selectbox("Building ID", sorted(df["building_id"]))
    row = df[df["building_id"] == bid]
    r = row.iloc[0]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(bid)
        st.markdown(
            f"**Use:** {r['primaryspaceusage']} / {r['sub_primaryspaceusage']}  \n"
            f"**Size:** {r['sqft']:,.0f} sq ft  \n"
            f"**Timezone:** {r['timezone']}  \n"
            f"**Estimated 2017 source EUI:** {r['source_eui_est']:.1f}  \n"
            f"**Actual label:** {core.LABELS[int(r['isEfficient'])]}"
        )
    with c2:
        show_probabilities(probs_for(row))
    with st.expander("Model inputs for this building"):
        details = row[core.INPUT_COLS].T.reset_index()
        details.columns = ["field", "value"]
        details["value"] = details["value"].astype(str)
        st.dataframe(details, hide_index=True)

# ----------------------------------------------------------------------------
# Batch prediction from a CSV
# ----------------------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with one building per row and these columns: "
             + ", ".join(f"`{c}`" for c in core.INPUT_COLS)
             + " (plus an optional `building_id`).")
    template = df.sample(min(5, len(df)), random_state=0)[["building_id"] + core.INPUT_COLS]
    st.download_button("Download a template CSV", template.to_csv(index=False),
                       "batch_template.csv", "text/csv")

    upload = st.file_uploader("Your CSV", type="csv")
    if upload is not None:
        raw = pd.read_csv(upload)
        missing = [c for c in core.INPUT_COLS if c not in raw.columns]
        if missing:
            st.warning("Missing columns treated as empty: " + ", ".join(missing))
        for msg in core.unknown_categories(raw):
            st.warning(f"Unseen category, ignored by the models: {msg}")
        result = core.predict_table(raw, MODELS)
        st.success(f"Predicted {len(result):,} buildings.")
        st.dataframe(result, hide_index=True)
        st.download_button("Download predictions", result.to_csv(index=False),
                           "predictions.csv", "text/csv", type="primary")
