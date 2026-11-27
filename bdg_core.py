"""
Building energy-efficiency app (Building Data Genome Project 2).

One file that holds everything the cloud server needs:
  * the preprocessing that turns raw building data into model features
  * inference with the trained models
  * the dashboard charts
  * the Gradio web UI (which also exposes a REST API)

It runs inside Colab (imported from the notebook) or on any server
(`python app.py`, e.g. a Hugging Face Space), as long as
model_bundle.joblib and buildings.csv sit next to it.
"""
import os
import tempfile
import functools

import numpy as np
import pandas as pd
import joblib
import sys
import matplotlib
if "ipykernel" not in sys.modules:   # on a server: no screen
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(HERE, "model_bundle.joblib")
DATA_PATH = os.path.join(HERE, "buildings.csv")
TEMPLATE_PATH = os.path.join(HERE, "batch_template.csv")

# ----------------------------------------------------------------------------
# Feature definitions (must match how final_data was built in the notebook)
# ----------------------------------------------------------------------------
NUMERIC_COLS = ["sqm", "sqft", "lat", "lng", "yearbuilt", "numberoffloors", "occupants"]
MONTHS = ["jan", "april", "july", "oct"]
METERS = ["electricity", "gas", "solar"]
METER_COLS = [f"{m}_{mo}" for m in METERS for mo in MONTHS]
CAT_COLS = ["primaryspaceusage", "sub_primaryspaceusage", "timezone"]
INPUT_COLS = NUMERIC_COLS + METER_COLS + CAT_COLS

MODEL_NAMES = [
    "Neural network",
    "Logistic regression",
    "Neural network (no meter data)",
    "Naive baseline",
]
LABELS = {1: "Efficient", 0: "Not efficient"}

# Colors: one fixed color per class and per model, used everywhere
EFF_COLOR, INEFF_COLOR = "#257de0", "#f47230"
MODEL_COLORS = dict(zip(MODEL_NAMES, ["#257de0", "#f47230", "#00b894", "#f2a600"]))
INK, INK_2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def fit_preprocessor(train_raw, all_raw):
    """
    Learns everything needed to turn raw rows into model features:
    mean/std of every numeric column (from the TRAINING rows only) and the
    list of one-hot columns (every category value seen in the data).
    """
    prep = {"mean": {}, "std": {}}
    for c in NUMERIC_COLS + METER_COLS:
        v = pd.to_numeric(train_raw[c], errors="coerce").fillna(0.0)
        std = float(v.std(ddof=0))
        prep["mean"][c] = float(v.mean())
        prep["std"][c] = std if std > 0 else 1.0
    prep["dummy_cols"] = [
        f"{cat}_{val}"
        for cat in CAT_COLS
        for val in sorted(all_raw[cat].dropna().astype(str).unique())
    ]
    prep["categories"] = {
        cat: sorted(all_raw[cat].dropna().astype(str).unique()) for cat in CAT_COLS
    }
    return prep


def make_features(raw, prep, use_meters=True):
    """
    raw: DataFrame with INPUT_COLS (numbers + category names as text).
    Returns the design matrix [intercept | standardized numeric | one-hot].
    Missing numbers become 0 (same as training); an unseen category gets all
    zeros in its one-hot group.
    """
    raw = raw.copy()
    for c in INPUT_COLS:
        if c not in raw.columns:
            raw[c] = np.nan
    num_cols = NUMERIC_COLS + (METER_COLS if use_meters else [])
    num = raw[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    mean = np.array([prep["mean"][c] for c in num_cols])
    std = np.array([prep["std"][c] for c in num_cols])
    num = (num - mean) / std

    col_index = {c: j for j, c in enumerate(prep["dummy_cols"])}
    dummies = np.zeros((len(raw), len(prep["dummy_cols"])))
    for cat in CAT_COLS:
        for r, val in enumerate(raw[cat].astype(str).to_numpy()):
            j = col_index.get(f"{cat}_{val}")
            if j is not None:
                dummies[r, j] = 1.0
    return np.column_stack([np.ones(len(raw)), num, dummies])


def feature_names(prep, use_meters=True):
    num_cols = NUMERIC_COLS + (METER_COLS if use_meters else [])
    return ["intercept"] + num_cols + prep["dummy_cols"]


# ----------------------------------------------------------------------------
# Loading artifacts
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@functools.lru_cache(maxsize=1)
def load_buildings():
    return pd.read_csv(DATA_PATH)


def reload_artifacts():
    load_bundle.cache_clear()
    load_buildings.cache_clear()


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def _class1_proba(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def predict_proba(raw, model_name="Neural network"):
    """Probability that each building in `raw` is efficient, for one model."""
    b = load_bundle()
    prep = b["prep"]
    if model_name == "Neural network":
        return _class1_proba(b["mlp"], make_features(raw, prep, True))
    if model_name == "Logistic regression":
        z = make_features(raw, prep, True) @ b["logistic_theta"]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    if model_name == "Neural network (no meter data)":
        return _class1_proba(b["mlp_no_meters"], make_features(raw, prep, False))
    if model_name == "Naive baseline":
        return np.ones(len(raw))
    raise ValueError(f"Unknown model: {model_name}")


def predict_table(raw, models=MODEL_NAMES):
    """One row per building: P(efficient) and the 0/1 label from every model."""
    out = pd.DataFrame(index=raw.index)
    if "building_id" in raw.columns:
        out["building_id"] = raw["building_id"]
    for m in models:
        p = predict_proba(raw, m)
        out[f"{m} | P(efficient)"] = np.round(p, 4)
        out[f"{m} | prediction"] = np.where(p > 0.5, LABELS[1], LABELS[0])
    return out.reset_index(drop=True)


def unknown_categories(raw):
    """Category values the models never saw (they get no one-hot signal)."""
    cats = load_bundle()["prep"]["categories"]
    msgs = []
    for c in CAT_COLS:
        if c in raw.columns:
            bad = sorted(set(raw[c].dropna().astype(str)) - set(cats[c]))
            if bad:
                msgs.append(f"{c}: {', '.join(bad[:5])}")
    return msgs


# ----------------------------------------------------------------------------
# Charts (each returns a matplotlib Figure)
# ----------------------------------------------------------------------------
def _finish(fig):
    """Tidy the layout and close the figure so it is not shown twice or kept in
    memory; a closed figure can still be displayed and saved."""
    fig.tight_layout()
    plt.close(fig)
    return fig


def fig_share_by_use(df):
    """How many buildings of each primary use are efficient vs not."""
    t = (df.groupby("primaryspaceusage")["isEfficient"]
           .agg(efficient="sum", total="count").sort_values("total"))
    if (t["total"] >= 5).any():          # hide tiny groups, unless that hides all
        t = t[t["total"] >= 5]
    t["not_efficient"] = t["total"] - t["efficient"]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.barh(y, t["efficient"], color=EFF_COLOR, height=0.7, label="Efficient")
    ax.barh(y, t["not_efficient"], left=t["efficient"], color=INEFF_COLOR,
            height=0.7, label="Not efficient", edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, t.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Number of buildings")
    ax.set_title("Buildings by primary use", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return _finish(fig)


def fig_eui_by_use(df):
    """Distribution of estimated 2017 source EUI for each primary use."""
    d = df[df["source_eui_est"] > 0]
    order = d.groupby("primaryspaceusage")["source_eui_est"].median().sort_values()
    big = d["primaryspaceusage"].value_counts().reindex(order.index) >= 5
    if big.any():
        order = order[big]
    if order.empty:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No buildings with energy data in this selection",
                ha="center", va="center", color=INK_2)
        return _finish(fig)
    data = [d.loc[d["primaryspaceusage"] == u, "source_eui_est"] for u in order.index]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1.4))
    ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#b6d5f8", edgecolor=EFF_COLOR),
               medianprops=dict(color=INK, linewidth=2),
               whiskerprops=dict(color=INK_2), capprops=dict(color=INK_2))
    ax.set_yticks(np.arange(1, len(order) + 1), order.index)
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Estimated source EUI (kWh per sq ft per year, log scale)")
    ax.set_title("Energy use intensity by primary use")
    return _finish(fig)


def fig_seasonal(df):
    """Median daily use per 1,000 sq ft in each season, efficient vs not."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = ["Jan", "Apr", "Jul", "Oct"]
    for ax, meter in zip(axes, ["electricity", "gas"]):
        for cls, color in [(1, EFF_COLOR), (0, INEFF_COLOR)]:
            sub = df[df["isEfficient"] == cls]
            cols = [f"{meter}_{mo}" for mo in MONTHS]
            per_area = sub[cols].div(sub["sqft"] / 1000, axis=0)
            per_area = per_area[(sub[cols].fillna(0) > 0).any(axis=1)]
            med = per_area.median()
            ax.plot(labels, med.to_numpy(), color=color, linewidth=2,
                    marker="o", markersize=7, label=LABELS[cls])
        ax.set_title(f"{meter.capitalize()}")
        ax.set_ylabel("kWh per day per 1,000 sq ft")
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("Typical daily use by season (2016, first week of the month)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    return _finish(fig)


def fig_site_share(df):
    """Share of efficient buildings at each data-collection site."""
    t = df.groupby("site")["isEfficient"].agg(["mean", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.barh(y, t["mean"] * 100, color=EFF_COLOR, height=0.7)
    for yi, (m, n) in enumerate(zip(t["mean"], t["count"])):
        ax.text(m * 100 + 1, yi, f"{m:.0%}  (n={n})", va="center", color=INK_2, fontsize=8)
    ax.set_yticks(y, t.index)
    ax.set_xlim(0, 118)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Buildings labeled efficient (%)")
    ax.set_title("Efficient buildings by site")
    return _finish(fig)


def fig_model_comparison(bundle=None):
    """Test-set metrics for every model (dot plot)."""
    b = bundle or load_bundle()
    metrics = pd.DataFrame(b["metrics"]).T.loc[MODEL_NAMES]
    cols = list(metrics.columns)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(cols))
    offsets = np.linspace(-0.27, 0.27, len(MODEL_NAMES))
    for off, m in zip(offsets, MODEL_NAMES):
        ax.scatter(metrics.loc[m, cols], y + off, s=55, color=MODEL_COLORS[m],
                   edgecolor="white", linewidth=1.5, label=m, zorder=3)
    ax.set_yticks(y, cols)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Score on the held-out test set")
    ax.set_title("Model comparison")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _finish(fig)


def fig_confusion(model_name="Neural network"):
    """Confusion matrix on the test set."""
    cm = np.array(load_bundle()["confusion"][model_name])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues", vmin=0)
    names = [LABELS[0], LABELS[1]]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if v > cm.max() / 2 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {model_name}", fontsize=11)
    return _finish(fig)


def fig_logistic_weights(top=15):
    """Largest logistic-regression weights (blue pushes toward 'efficient')."""
    b = load_bundle()
    w = pd.Series(b["logistic_theta"], index=feature_names(b["prep"], True)).drop("intercept")
    w = w.reindex(w.abs().sort_values().index[-top:])
    fig, ax = plt.subplots(figsize=(7, 0.3 * top + 1.4))
    ax.barh(np.arange(len(w)), w, color=[EFF_COLOR if v > 0 else INEFF_COLOR for v in w],
            height=0.7)
    ax.set_yticks(np.arange(len(w)), [n.replace("sub_primaryspaceusage_", "sub: ")
                                      .replace("primaryspaceusage_", "use: ") for n in w.index])
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Weight (positive = more likely efficient)")
    ax.set_title(f"Logistic regression: top {top} weights")
    return _finish(fig)


def fig_prediction(probs):
    """P(efficient) from each model for one building, with the 0.5 threshold."""
    names = list(probs.keys())
    vals = [probs[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, vals, color=[MODEL_COLORS[n] for n in names], height=0.6)
    for yi, v in zip(y, vals):
        ax.text(min(v + 0.02, 0.9), yi, f"{v:.0%}", va="center", color=INK, fontsize=9)
    ax.axvline(0.5, color=INK_2, linestyle="--", linewidth=1)
    ax.text(0.5, -0.75, " threshold", color=INK_2, fontsize=8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Probability the building is efficient")
    ax.set_title("Prediction by model")
    return _finish(fig)


# ----------------------------------------------------------------------------
# Functions behind the UI / API endpoints
# ----------------------------------------------------------------------------
def predict_single(*values):
    """values arrive in INPUT_COLS order. Returns (label dict, chart, table, note)."""
    raw = pd.DataFrame([dict(zip(INPUT_COLS, values))])
    probs = {m: float(predict_proba(raw, m)[0]) for m in MODEL_NAMES[:3]}
    p = probs["Neural network"]
    note = ""
    meters = raw[METER_COLS].iloc[0].apply(pd.to_numeric, errors="coerce").fillna(0)
    if (meters == 0).all():
        note = ("No meter readings were entered, so trust "
                "**Neural network (no meter data)** most for this building. ")
    bad = unknown_categories(raw)
    if bad:
        note += "Unseen categories (ignored by the models): " + "; ".join(bad)
    return ({LABELS[1]: p, LABELS[0]: 1 - p}, fig_prediction(probs),
            predict_table(raw, MODEL_NAMES[:3]).T.reset_index()
            .rename(columns={"index": "field", 0: "value"}), note)


def lookup_building(building_id):
    """Predictions for a building already in the dataset, next to its true label."""
    df = load_buildings()
    row = df[df["building_id"] == building_id]
    if row.empty:
        return f"Building '{building_id}' not found.", None, None
    r = row.iloc[0]
    probs = {m: float(predict_proba(row, m)[0]) for m in MODEL_NAMES[:3]}
    summary = (
        f"### {building_id}\n"
        f"**Use:** {r['primaryspaceusage']} / {r['sub_primaryspaceusage']}  \n"
        f"**Size:** {r['sqft']:,.0f} sq ft  ·  **Timezone:** {r['timezone']}  \n"
        f"**Estimated 2017 source EUI:** {r['source_eui_est']:.1f}  \n"
        f"**Actual label:** {LABELS[int(r['isEfficient'])]}"
    )
    details = row[INPUT_COLS].T.reset_index()
    details.columns = ["field", "value"]
    details["value"] = details["value"].astype(str)
    return summary, fig_prediction(probs), details


def predict_batch(file):
    """CSV in (one building per row, INPUT_COLS) -> table + downloadable CSV out."""
    if file is None:
        return None, None, "Upload a CSV first."
    path = file if isinstance(file, str) else file.name
    raw = pd.read_csv(path)
    missing = [c for c in INPUT_COLS if c not in raw.columns]
    result = predict_table(raw, MODEL_NAMES[:3])
    out_path = os.path.join(tempfile.mkdtemp(), "predictions.csv")
    result.to_csv(out_path, index=False)
    msg = f"Predicted {len(raw)} buildings."
    if missing:
        msg += f" Missing columns treated as empty: {', '.join(missing)}."
    bad = unknown_categories(raw)
    if bad:
        msg += " Unseen categories: " + "; ".join(bad)
    return result, out_path, msg


def write_batch_template(n=5):
    df = load_buildings()
    df.sample(min(n, len(df)), random_state=0)[["building_id"] + INPUT_COLS] \
        .to_csv(TEMPLATE_PATH, index=False)
    return TEMPLATE_PATH


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------
def build_demo():
    import gradio as gr

    df = load_buildings()
    b = load_bundle()
    cats = b["prep"]["categories"]
    med = df[NUMERIC_COLS + METER_COLS].median(numeric_only=True)
    m = b["metrics"]["Neural network"]

    with gr.Blocks(title="Building Energy Efficiency") as demo:
        gr.Markdown(
            "# Building energy efficiency\n"
            "Predicts whether a building uses less energy per square foot than the "
            "national median for its type (Building Data Genome Project 2). "
            f"Dataset: **{len(df):,} buildings** at **{df['site'].nunique()} sites**, "
            f"**{df['isEfficient'].mean():.0%}** labeled efficient. "
            f"Best model test accuracy: **{m['Accuracy']:.1%}**."
        )

        with gr.Tab("Dashboard"):
            with gr.Row():
                gr.Plot(fig_share_by_use(df))
                gr.Plot(fig_eui_by_use(df))
            with gr.Row():
                gr.Plot(fig_seasonal(df))
            with gr.Row():
                gr.Plot(fig_site_share(df))
                gr.Plot(fig_logistic_weights())
            gr.Markdown("## Model performance")
            gr.Plot(fig_model_comparison(b))
            with gr.Row():
                cm_model = gr.Dropdown(MODEL_NAMES, value="Neural network",
                                       label="Confusion matrix for")
            cm_plot = gr.Plot(fig_confusion("Neural network"))
            cm_model.change(fig_confusion, cm_model, cm_plot, api_name=False)

        with gr.Tab("Predict a new building"):
            gr.Markdown("Fill in what you know; blank numbers are treated as missing "
                        "(0), the same way the training data was.")
            comps = {}
            with gr.Row():
                comps["primaryspaceusage"] = gr.Dropdown(
                    cats["primaryspaceusage"], value="Education", label="Primary use")
                comps["sub_primaryspaceusage"] = gr.Dropdown(
                    cats["sub_primaryspaceusage"], value=cats["sub_primaryspaceusage"][0],
                    label="Sub-use")
                comps["timezone"] = gr.Dropdown(
                    cats["timezone"], value="US/Eastern", label="Timezone")
            with gr.Row():
                for c in NUMERIC_COLS:
                    comps[c] = gr.Number(value=None if pd.isna(med.get(c)) else
                                         round(float(med[c]), 2), label=c)
            with gr.Accordion("Average daily meter readings, kWh (first week of month)",
                              open=True):
                for meter in METERS:
                    with gr.Row():
                        for mo in MONTHS:
                            c = f"{meter}_{mo}"
                            comps[c] = gr.Number(value=round(float(med.get(c, 0) or 0), 2),
                                                 label=c)
            btn = gr.Button("Predict", variant="primary")
            with gr.Row():
                out_label = gr.Label(label="Neural network")
                out_plot = gr.Plot()
            out_note = gr.Markdown()
            out_table = gr.Dataframe(label="All models")
            btn.click(predict_single, [comps[c] for c in INPUT_COLS],
                      [out_label, out_plot, out_table, out_note], api_name="predict")

        with gr.Tab("Look up a building"):
            ids = sorted(df["building_id"].tolist())
            bid = gr.Dropdown(ids, value=ids[0], label="Building ID", filterable=True)
            look_btn = gr.Button("Show prediction", variant="primary")
            with gr.Row():
                look_md = gr.Markdown()
                look_plot = gr.Plot()
            look_table = gr.Dataframe(label="Model inputs")
            look_btn.click(lookup_building, bid, [look_md, look_plot, look_table],
                           api_name="lookup")

        with gr.Tab("Batch prediction (CSV)"):
            gr.Markdown("Upload a CSV with one building per row and these columns: `"
                        + "`, `".join(INPUT_COLS) + "` (plus an optional `building_id`).")
            gr.File(value=write_batch_template(), label="Download a template")
            up = gr.File(label="Your CSV", file_types=[".csv"], type="filepath")
            batch_btn = gr.Button("Predict all", variant="primary")
            batch_msg = gr.Markdown()
            batch_table = gr.Dataframe()
            batch_file = gr.File(label="Download predictions")
            batch_btn.click(predict_batch, up, [batch_table, batch_file, batch_msg],
                            api_name="predict_batch")
    return demo


if __name__ == "__main__":
    # On a server: python app.py  (Hugging Face Spaces runs this automatically)
    build_demo().launch(server_name="0.0.0.0",
                        server_port=int(os.environ.get("PORT", 7860)),
                        allowed_paths=[HERE])
