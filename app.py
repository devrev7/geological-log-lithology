"""Streamlit app: estimate lithology from five well-log readings.

Run with:  streamlit run app.py
"""

import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

CURVES = ["RHOB", "GR", "NPHI", "PEF", "DTC"]
UNITS = {"RHOB": "g/cc", "GR": "API", "NPHI": "v/v", "PEF": "b/e", "DTC": "us/ft"}
RANGES = {  # (min, max, default, step) — limits are the physical ones used in cleaning
    "RHOB": (1.5, 3.0, 2.20, 0.01),
    "GR": (0.0, 250.0, 60.0, 1.0),
    "NPHI": (0.0, 1.0, 0.35, 0.01),
    "PEF": (0.0, 10.0, 3.50, 0.1),
    "DTC": (40.0, 190.0, 110.0, 1.0),
}
COLORS = {"Soft shale": "#8c6d46", "Limestone / chalk": "#2a78d6", "Compacted shale": "#3a9e5f"}
PRESETS = {
    "Soft shale (upper well)": (2.01, 68.0, 0.50, 2.6, 146.0),
    "Limestone / chalk": (2.55, 16.0, 0.17, 4.8, 70.0),
    "Compacted shale (deep)": (2.48, 96.0, 0.31, 4.4, 88.0),
    "Between two rocks": (2.25, 55.0, 0.35, 3.8, 110.0),
}


@st.cache_resource
def load_model():
    saved = joblib.load("lithology_model.joblib")
    return saved["model"], saved["curves"], list(saved["classes"])


st.set_page_config(page_title="Lithology estimator", page_icon="🪨", layout="wide")
st.title("🪨 Lithology estimator")
st.caption(
    "Estimates rock type from five wireline readings. Trained on clusters interpreted from a single "
    "FORCE2020 well (1138.7–2993.9 m), so it knows three rock types and reproduces that interpretation — "
    "it is not validated against measured lithology."
)

model, curves, classes = load_model()

tab_single, tab_file = st.tabs(["One reading", "Whole log (CSV)"])

with tab_single:
    left, right = st.columns([1, 1.3])

    with left:
        preset = st.selectbox("Start from an example", ["Type my own"] + list(PRESETS))
        defaults = PRESETS.get(preset)

        values = {}
        for i, curve in enumerate(CURVES):
            low, high, default, stepsize = RANGES[curve]
            values[curve] = st.slider(
                f"{curve} ({UNITS[curve]})",
                min_value=low,
                max_value=high,
                value=float(defaults[i]) if defaults else default,
                step=stepsize,
            )

    reading = pd.DataFrame([[values[c] for c in curves]], columns=curves)
    probabilities = pd.Series(model.predict_proba(reading)[0], index=classes).sort_values(ascending=False)
    best = probabilities.index[0]

    with right:
        st.subheader(best)
        st.metric("Confidence", f"{probabilities.iloc[0]:.0%}")

        for name, value in probabilities.items():
            st.write(f"**{name}** — {value:.1%}")
            st.progress(float(value))

        if probabilities.iloc[0] < 0.7:
            st.warning(
                "Low confidence. Readings between two rock types sit near a layer boundary, "
                "where the model is least reliable."
            )

        st.caption(
            "Curves by influence: DTC 34%, NPHI 30%, GR 21%, RHOB 13%, PEF 2%. "
            "Accuracy 97.7% against the clustering (depth-blocked cross-validation)."
        )

with tab_file:
    st.write(
        "Upload a CSV with columns **RHOB, GR, NPHI, PEF, DTC** (a DEPTH_MD column is used for the plot "
        "if present). Rows with a blank reading are skipped."
    )
    uploaded = st.file_uploader("CSV file", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [c for c in curves if c not in data.columns]

        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}")
        else:
            usable = data.dropna(subset=curves).copy()
            usable["lithology"] = model.predict(usable[curves])
            probs = model.predict_proba(usable[curves])
            usable["confidence"] = probs.max(axis=1).round(3)

            st.success(f"Labelled {len(usable):,} of {len(data):,} rows.")
            counts = usable["lithology"].value_counts()
            st.bar_chart(counts)

            if "DEPTH_MD" in usable.columns:
                fig, ax = plt.subplots(figsize=(3.5, 9))
                for name, color in COLORS.items():
                    ax.fill_betweenx(usable["DEPTH_MD"], 0, 1, where=usable["lithology"] == name,
                                     color=color, label=name)
                ax.set_xticks([])
                ax.set_ylabel("Depth (m)")
                ax.invert_yaxis()
                ax.legend(loc="lower left", bbox_to_anchor=(1, 0), frameon=False)
                ax.set_title("Estimated lithology")
                plt.tight_layout()
                st.pyplot(fig)

            st.dataframe(usable.head(50))

            buffer = io.StringIO()
            usable.to_csv(buffer, index=False)
            st.download_button("Download labelled CSV", buffer.getvalue(),
                               file_name="lithology_predictions.csv", mime="text/csv")
