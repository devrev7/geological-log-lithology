# Lithology estimator — single-well log analysis

Estimates rock type from five wireline log readings, using an interpretation built from one unlabeled
FORCE2020 well (1138.7–2993.9 m).

**Live app:** enter RHOB, GR, NPHI, PEF and DTC and get probabilities for three rock types, or upload a CSV to
label a whole log.

## How it was built

1. **EDA** (`Data_Analysis_eda.ipynb`) — curve behaviour, correlations, depth zones, missing data.
2. **Cleaning** (`Data_Cleaning.ipynb`) — removed readings no rock can produce: a washed-out shallow zone
   (RHOB < 1.5 with three tools frozen), PEF above 10 (barite mud), and 208 PEF spikes. 3,337 values changed,
   every change flagged in the saved file. → `force2020_cleaned.csv`
3. **Feature trials** (`feature_engineering_beta.ipynb`, `feature_engineering_beta_cleaned.ipynb`) — windowing
   and four clustering methods, before and after cleaning.
4. **Feature testing** (`feature_testing.ipynb`) — median vs mean, window 2–8 m, with/without spread columns,
   DTC handling, and number of clusters. Winner: **4 m rolling median of five curves, RobustScaler, k = 3**.
5. **Final model** (`feature_engineering_final.ipynb`) — KMeans k = 3 on 12,104 depths, clusters named from
   their readings. → `force2020_lithology.csv`
6. **Predictor** (`lithology_predictor.ipynb`) — Random Forest trained on those labels, validated with
   depth-blocked cross-validation. → `lithology_model.joblib`
7. **App** (`app.py`) — Streamlit front end.

## Results

| Estimated rock | RHOB | GR | NPHI | PEF | DTC | Main interval |
|---|---|---|---|---|---|---|
| Soft shale | 2.01 | 65 | 0.50 | 2.8 | 145 | 1138.7–2394.8 m |
| Limestone / chalk | 2.54 | 18 | 0.18 | 4.7 | 71 | 2439.5–2732.3 m |
| Compacted shale | 2.48 | 96 | 0.31 | 4.4 | 88 | 2790.2–2993.9 m |

The boundaries near 2395 m and 2741 m were found by KMeans, Gaussian Mixture, HDBSCAN and Agglomerative
clustering alike, before and after cleaning, at every window size from 2 to 8 m.

Predictor accuracy: **97.7%** on depth-blocked cross-validation (worst block 93.6%). Splitting by 50 m depth
blocks matters — readings 0.152 m apart are nearly identical, so a shuffled split leaks and reports 98.9%.

## Limitations

- The dataset has **no lithology labels**, so the rock names are an interpretation of log behaviour and cannot
  be checked for accuracy. The predictor reproduces that interpretation; it does not verify it.
- **One well**, so nothing can be cross-checked against a neighbour.
- **No caliper (CALI) or DRHO**, so borehole problems can only be inferred from the curves themselves.
- Only the **three rock types present in this well** are known to the model. A sandstone or coal would still be
  forced into one of them.
- Nothing above 1138.7 m or below 2993.9 m can be classified: NPHI and PEF were not recorded there.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data: FORCE 2020 well log subset (Kaggle, "geological logging dataset").
