
"""Silica concentrate prediction: an honest look at what the plant data supports.

Reads two small CSVs exported from the notebook:
  silica_results.csv   - hourly test-period actuals and each model's prediction
  rf_importances.csv   - feature importances of the "sensors + last hour" model
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Silica prediction, honestly", layout="wide")

LEAKY_R2 = 0.998  # first notebook version: random split + % Iron Concentrate as an input

# Display names, in the order they appear everywhere in the app
MODELS = {
    "mean_baseline": "Predict the average",
    "rf_sensors_only": "Random Forest, sensors only",
    "repeat_last_hour": "Repeat last hour's assay",
    "rf_sensors_plus_last_hour": "Random Forest, sensors + last hour",
}
COLOURS = {
    "Actual (lab assay)": "#1C2326",
    "Predict the average": "#A7B3B9",
    "Random Forest, sensors only": "#B8862B",
    "Repeat last hour's assay": "#4F7A94",
    "Random Forest, sensors + last hour": "#6E2A1E",
}


@st.cache_data
def load_results():
    df = pd.read_csv("silica_results.csv", parse_dates=["date"]).set_index("date")
    return df


@st.cache_data
def load_importances():
    return pd.read_csv("rf_importances.csv")


def score(actual, predicted):
    err = actual - predicted
    r2 = 1 - (err**2).sum() / ((actual - actual.mean()) ** 2).sum()
    rmse = float(np.sqrt((err**2).mean()))
    return r2, rmse


results = load_results()
scores = pd.DataFrame(
    [(MODELS[col], *score(results["actual"], results[col])) for col in MODELS],
    columns=["Model", "R²", "RMSE"],
)
best = scores.set_index("Model")

# ---------------------------------------------------------------- headline
st.title("My first silica model scored R² 0.998. That number was wrong.")
st.write(
    "This project predicts % silica in iron ore flotation concentrate from plant data. "
    "The first version looked near-perfect because of data leakage. "
    "This app shows the rebuilt, leak-free results and what they say about the plant."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("First version (leaky)", f"{LEAKY_R2:.3f}")
c2.metric("Sensors only", f"{best.loc[MODELS['rf_sensors_only'], 'R²']:.3f}")
c3.metric("Repeat last hour", f"{best.loc[MODELS['repeat_last_hour'], 'R²']:.3f}")
c4.metric("Sensors + last hour", f"{best.loc[MODELS['rf_sensors_plus_last_hour'], 'R²']:.3f}")
st.caption("R² on the same held-out test hours. 1 is perfect; 0 is no better than predicting the average.")

# ---------------------------------------------------------------- the leak
st.header("Where the 0.998 came from")
left, right = st.columns(2)
left.markdown(
    "**A same-assay input.** % Iron Concentrate is measured in the same lab test as silica. "
    "The plant doesn't have it before the silica result, so the model was reading the answer's twin."
)
right.markdown(
    "**A random split of 20-second rows.** Each lab hour has about 180 near-identical rows. "
    "Shuffling put rows from the same hour in both training and test, so the model recognised hours "
    "instead of learning the process."
)
st.markdown(
    "The fix: drop % Iron Concentrate, average to one row per lab hour, "
    "and test only on later hours the model has never seen."
)

# ---------------------------------------------------------------- comparison
st.header("How the honest models compare")
metric = st.radio("Measure", ["RMSE (lower is better)", "R² (higher is better)"], horizontal=True)
col = "RMSE" if metric.startswith("RMSE") else "R²"
bars = (
    alt.Chart(scores)
    .mark_bar()
    .encode(
        x=alt.X(f"{col}:Q", title=col if col == "R²" else "RMSE (% silica)"),
        y=alt.Y("Model:N", sort=list(MODELS.values()), title=None),
        color=alt.Color("Model:N", scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())), legend=None),
        tooltip=["Model", alt.Tooltip("R²:Q", format=".3f"), alt.Tooltip("RMSE:Q", format=".3f")],
    )
    .properties(height=220)
)
st.altair_chart(bars, width="stretch")

# ---------------------------------------------------------------- time series
st.header("Actual vs predicted, hour by hour")
chosen = st.multiselect(
    "Models to show",
    list(MODELS.values()),
    default=["Repeat last hour's assay", "Random Forest, sensors + last hour"],
)
start, end = results.index.min().to_pydatetime(), results.index.max().to_pydatetime()
window = st.slider("Date range", min_value=start, max_value=end, value=(start, end), format="YYYY-MM-DD")

view = results.loc[window[0]:window[1]].rename(columns=MODELS | {"actual": "Actual (lab assay)"})
long = view[["Actual (lab assay)", *chosen]].reset_index().melt("date", var_name="Series", value_name="% silica")
lines = (
    alt.Chart(long)
    .mark_line(strokeWidth=1.4)
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("% silica:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.condition(alt.datum.Series == "Actual (lab assay)", alt.value([1, 0]), alt.value([5, 3])),
        tooltip=["date:T", "Series:N", alt.Tooltip("% silica:Q", format=".2f")],
    )
    .properties(height=360)
    .interactive(bind_y=False)
)
st.altair_chart(lines, width="stretch")
st.caption("Solid line: the lab result. Dashed lines: predictions. Scroll to zoom in time.")

# ---------------------------------------------------------------- importances
st.header("What the best model relies on")
imp = load_importances()
imp_chart = (
    alt.Chart(imp.head(10))
    .mark_bar(color="#6E2A1E")
    .encode(
        x=alt.X("importance:Q", title="Share of the model's decisions"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
    )
    .properties(height=280)
)
st.altair_chart(imp_chart, width="stretch")

# ---------------------------------------------------------------- takeaway
st.header("What this means for the plant")
st.markdown(
    "- Hourly sensor readings on their own explain very little of the change in silica.\n"
    "- Silica moves slowly, so last hour's assay is the strongest predictor of this hour's.\n"
    "- Adding sensors to that assay doesn't improve on simply repeating it.\n"
    "- Next steps: sensor trends within the hour, predicting the hour-to-hour change, "
    "and allowing for the lab's real reporting delay."
)

with st.expander("Method details"):
    st.markdown(
        f"- Data: public iron ore flotation plant dataset (Mar–Sep 2017), 20-second sensor rows and hourly lab assays.\n"
        f"- Rows averaged to one per lab hour; hours with a data gap before them dropped from the lag feature.\n"
        f"- Chronological split: first 80% of hours for training, last 20% ({len(results)} hours) for testing.\n"
        f"- Random Forest: 300 trees, at least 5 hours per leaf.\n"
        f"- Scores are computed live in this app from the exported test-period predictions."
    )