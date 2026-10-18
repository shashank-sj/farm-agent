"""
Train the yield model used by src/tools/yield_tool.py.

Data: "Agricultural Crop Yield in Indian States" (Kaggle:
akshatgupta7/crop-yield-in-indian-states-dataset, CC BY-SA 4.0), 1997–2020.
Columns: Crop, Crop_Year, Season, State, Area (ha), Production (t),
Annual_Rainfall (mm), Fertilizer (kg), Pesticide (kg), Yield (production/area).

Choices worth defending:
  * Target = log1p(Yield). Yields span 0.1 t/ha (pulses) to 80+ t/ha
    (sugarcane); on the raw scale the loss is dominated by a few crops.
  * Production is DROPPED — Yield = Production / Area, so keeping it is
    target leakage (a perfect score that means nothing).
  * Fertilizer and Pesticide are converted to per-hectare rates. As totals they
    mostly re-encode Area, which hides the actual input intensity.
  * Coconut is dropped: its "production" is counted in nuts, not tonnes.
  * Split by TIME (train ≤ 2015, test ≥ 2016), not randomly. The tool is used
    to predict a future season; a random split lets the model see the same
    state-crop in neighbouring years and overstates accuracy. The script also
    reports the random-split score so the gap is visible.
  * Baseline = median train yield for the same (crop, state, season), falling
    back to crop median. A model that can't beat a lookup table isn't worth shipping.

Usage:
    python -m scripts.train_yield_model --csv /path/to/crop_yield.csv --out outputs/yield_model
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL = ["Crop", "Season", "State"]
NUMERIC = ["log_area", "fertilizer_per_ha", "pesticide_per_ha", "Annual_Rainfall"]
FEATURE_ORDER = CATEGORICAL + NUMERIC
EXCLUDED_CROPS = {"Coconut"}
TEST_FROM_YEAR = 2016


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()     # Season values carry trailing spaces
    n_raw = len(df)
    df = df[~df["Crop"].isin(EXCLUDED_CROPS)]
    df = df[(df["Area"] > 0) & (df["Yield"] > 0)]
    df = df.dropna(subset=["Annual_Rainfall", "Fertilizer", "Pesticide"])
    # Drop extreme yields per crop (data-entry errors), using the 99.5th percentile.
    cap = df.groupby("Crop")["Yield"].transform(lambda s: s.quantile(0.995))
    df = df[df["Yield"] <= cap].copy()

    df["log_area"] = np.log1p(df["Area"])
    df["fertilizer_per_ha"] = df["Fertilizer"] / df["Area"]
    df["pesticide_per_ha"] = df["Pesticide"] / df["Area"]
    df["target"] = np.log1p(df["Yield"])
    print(f"rows: {n_raw} raw → {len(df)} after cleaning | years {df.Crop_Year.min()}–{df.Crop_Year.max()}")
    return df


def make_model() -> Pipeline:
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # Native categorical splits: trees split on sets of categories, which beats
    # one-hot for ~55 crops × ~30 states. Negative codes (unseen) are treated as missing.
    gbm = HistGradientBoostingRegressor(
        categorical_features=[0, 1, 2],
        learning_rate=0.05,
        max_iter=3000,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
    )
    return Pipeline([("encode", encode), ("gbm", gbm)])


def evaluate(y_log_true, y_log_pred) -> dict:
    y_true, y_pred = np.expm1(y_log_true), np.expm1(y_log_pred)
    ape = np.abs(y_pred - y_true) / np.maximum(y_true, 1e-6)
    return {
        "r2_log": round(float(r2_score(y_log_true, y_log_pred)), 4),
        "r2_raw": round(float(r2_score(y_true, y_pred)), 4),
        "mae_t_per_ha": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "median_ape_pct": round(float(np.median(ape) * 100), 2),
    }


def lookup_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    keyed = train.groupby(CATEGORICAL)["target"].median()
    by_crop = train.groupby("Crop")["target"].median()
    overall = train["target"].median()
    preds = []
    for _, row in test.iterrows():
        key = (row["Crop"], row["Season"], row["State"])
        preds.append(keyed.get(key, by_crop.get(row["Crop"], overall)))
    return np.asarray(preds)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/yield_model")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(args.csv)
    print(f"corr(fertilizer_per_ha, year) = {df['fertilizer_per_ha'].corr(df['Crop_Year']):.3f}  "
          "(close to 1.0 would mean the per-ha rate is mostly a year trend, not a farm input)")

    train, test = df[df.Crop_Year < TEST_FROM_YEAR], df[df.Crop_Year >= TEST_FROM_YEAR]
    print(f"time split: train {len(train)} rows (≤{TEST_FROM_YEAR - 1}), test {len(test)} rows (≥{TEST_FROM_YEAR})")

    model = make_model().fit(train[FEATURE_ORDER], train["target"])
    metrics = {
        "test": evaluate(test["target"], model.predict(test[FEATURE_ORDER])),
        "baseline_lookup_test": evaluate(test["target"], lookup_baseline(train, test)),
    }

    # Same model, random split — to show how optimistic a random split would be.
    r_train, r_test = train_test_split(df, test_size=len(test) / len(df), random_state=42)
    random_model = make_model().fit(r_train[FEATURE_ORDER], r_train["target"])
    metrics["random_split_test"] = evaluate(r_test["target"], random_model.predict(r_test[FEATURE_ORDER]))

    # Per-crop error on the time-split test set, for the crops that matter most
    per_crop = {}
    test_pred = model.predict(test[FEATURE_ORDER])
    for crop, idx in test.groupby("Crop").groups.items():
        mask = test.index.isin(idx)
        if mask.sum() >= 30:
            per_crop[crop] = evaluate(test["target"][mask], test_pred[mask])["median_ape_pct"]
    metrics["test_median_ape_pct_by_crop"] = dict(sorted(per_crop.items(), key=lambda kv: kv[1]))

    print(json.dumps(metrics, indent=2))

    # Refit on ALL years for serving, keeping the iteration count the time-split
    # model chose, so the deployed model also knows the most recent seasons.
    final = make_model()
    final.set_params(gbm__early_stopping=False, gbm__max_iter=int(model.named_steps["gbm"].n_iter_))
    final.fit(df[FEATURE_ORDER], df["target"])
    joblib.dump(final, out / "yield_model.joblib")

    metadata = {
        "feature_order": FEATURE_ORDER,
        "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
        "state_rainfall_median": df.groupby("State")["Annual_Rainfall"].median().round(1).to_dict(),
        "global_rainfall_median": round(float(df["Annual_Rainfall"].median()), 1),
        "target": "log1p(yield, tonnes/hectare)",
        "split": f"time-based: train < {TEST_FROM_YEAR}, test >= {TEST_FROM_YEAR}",
        "n_rows": int(len(df)),
        "n_iter": int(model.named_steps["gbm"].n_iter_),
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {out / 'yield_model.joblib'} and metadata.json")


if __name__ == "__main__":
    main()
