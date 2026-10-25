"""Train the scorecard and the two benchmark models, and persist results.

Pipeline
--------
1. Load the German Credit data (downloaded on first use).
2. Stratified 70/30 train/test split (the dataset has only 1,000 rows,
   so 5-fold cross-validation AUCs on the training set are reported too).
3. Fit three models on the training set:

   - **Scorecard** -- WOE binning + logistic regression, scaled to a
     600-base / PDO-20 points system (see :mod:`src.scorecard`).
   - **Plain logistic regression** -- one-hot + scaled numerics.
   - **XGBoost** -- gradient boosting on the same encoding, as a
     "black-box ceiling" to show what the interpretable scorecard
     gives up (usually very little).

4. Derive approve/review/decline score bands from the training-set
   cutoff tradeoff, evaluate everything on the held-out test set, save
   ``reports/metrics.json`` and the scorecard bundle to
   ``models/scorecard.joblib`` (git-ignored, rebuilt by this script).

Run with::

    python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    load_data,
)
from src.evaluate import (
    calibration_data,
    confusion_at_cutoff,
    cutoff_tradeoff,
    model_metrics,
)
from src.scorecard import Scorecard

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "scorecard.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
RANDOM_STATE = 42

# Decision-band policy, expressed as modelled default-probability
# targets (standard practice: bands are set on PD, the score follows).
APPROVE_MAX_PD = 0.15  # approve if modelled PD is at most 15%
DECLINE_MIN_PD = 0.50  # decline if modelled PD is at least 50%


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def make_logreg() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def make_xgboost() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def score_for_pd(card: Scorecard, pd: float) -> float:
    """Score at which the scorecard's modelled PD equals ``pd``.

    Inverts the scaling: score = offset + factor * ln(odds of good),
    and odds of good = (1 - PD) / PD.
    """
    return float(card.offset + card.factor * np.log((1.0 - pd) / pd))


def derive_bands(card: Scorecard) -> tuple[float, float]:
    """Approve/decline score cutoffs from the PD policy targets.

    Bands are set on modelled default probability (approve if
    PD <= ``APPROVE_MAX_PD``, decline if PD >= ``DECLINE_MIN_PD``,
    manual review in between) and converted to score cutoffs through
    the scorecard's own scaling, so the policy is consistent by
    construction. The realized bad rates of each band are reported in
    the README / metrics from the held-out test set.
    """
    return score_for_pd(card, APPROVE_MAX_PD), score_for_pd(card, DECLINE_MIN_PD)


def cv_auc(make_model, x: pd.DataFrame, y: pd.Series) -> dict:
    """5-fold cross-validated ROC-AUC (out-of-fold predictions)."""
    from sklearn.metrics import roc_auc_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(make_model(), x, y, cv=cv, method="predict_proba")[:, 1]
    return {"roc_auc": float(roc_auc_score(y, proba))}


def cv_auc_scorecard(df: pd.DataFrame) -> dict:
    """5-fold CV AUC for the scorecard via a manual fold loop."""
    from sklearn.metrics import roc_auc_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = np.zeros(len(df))
    for fit_idx, val_idx in cv.split(df, df["default"]):
        card = Scorecard(CATEGORICAL_FEATURES, NUMERIC_FEATURES).fit(
            df.iloc[fit_idx]
        )
        proba[val_idx] = card.predict_default_proba(df.iloc[val_idx])
    return {"roc_auc": float(roc_auc_score(df["default"], proba))}


def train_and_save(df: pd.DataFrame | None = None) -> dict:
    """Train all models, write metrics + bundle, and return the bundle."""
    if df is None:
        df = load_data()

    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["default"]
    )
    y_train = train_df["default"].to_numpy()
    y_test = test_df["default"].to_numpy()
    x_train, x_test = train_df[FEATURES], test_df[FEATURES]

    # --- models ---------------------------------------------------------
    scorecard = Scorecard(CATEGORICAL_FEATURES, NUMERIC_FEATURES).fit(train_df)
    logreg = make_logreg().fit(x_train, y_train)
    xgboost = make_xgboost().fit(x_train, y_train)

    probas = {
        "scorecard": scorecard.predict_default_proba(test_df),
        "logistic_regression": logreg.predict_proba(x_test)[:, 1],
        "xgboost": xgboost.predict_proba(x_test)[:, 1],
    }
    test_metrics = {name: model_metrics(y_test, p) for name, p in probas.items()}

    cv = {
        "scorecard": cv_auc_scorecard(train_df),
        "logistic_regression": cv_auc(make_logreg, x_train, train_df["default"]),
        "xgboost": cv_auc(make_xgboost, x_train, train_df["default"]),
    }

    # --- bands + scorecard-specific analysis ---------------------------
    test_scores = scorecard.score(test_df)
    approve_cutoff, decline_cutoff = derive_bands(scorecard)

    tradeoff_cutoffs = sorted(
        {decline_cutoff, approve_cutoff}
        | set(np.quantile(test_scores, np.linspace(0.05, 0.95, 10)).tolist())
    )
    tradeoff = cutoff_tradeoff(y_test, test_scores, [round(c, 1) for c in tradeoff_cutoffs])

    band_masks = {
        "approve": test_scores >= approve_cutoff,
        "review": (test_scores >= decline_cutoff) & (test_scores < approve_cutoff),
        "decline": test_scores < decline_cutoff,
    }
    band_performance = {
        band: {
            "count": int(mask.sum()),
            "share": float(mask.mean()),
            "bad_rate": float(y_test[mask].mean()) if mask.sum() else 0.0,
        }
        for band, mask in band_masks.items()
    }

    metrics = {
        "dataset": {
            "name": "UCI Statlog German Credit",
            "rows": int(len(df)),
            "defaults": int(df["default"].sum()),
            "default_rate": float(df["default"].mean()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_defaults": int(y_test.sum()),
        },
        "models_test": test_metrics,
        "models_cv_train_auc": cv,
        "scorecard": {
            "base_score": scorecard.base_score,
            "base_odds": scorecard.base_odds,
            "pdo": scorecard.pdo,
            "approve_cutoff": round(approve_cutoff, 1),
            "decline_cutoff": round(decline_cutoff, 1),
            "information_values": scorecard.iv_table.to_dict(orient="records")
            if scorecard.iv_table is not None
            else [],
            "calibration_test": calibration_data(y_test, probas["scorecard"]),
            "band_policy": {
                "approve_max_pd": APPROVE_MAX_PD,
                "decline_min_pd": DECLINE_MIN_PD,
            },
            "band_performance_test": band_performance,
            "confusion_at_approve_cutoff": confusion_at_cutoff(
                y_test, test_scores, approve_cutoff
            ),
            "cutoff_tradeoff_test": tradeoff.to_dict(orient="records"),
        },
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = {
        "scorecard": scorecard,
        "approve_cutoff": approve_cutoff,
        "decline_cutoff": decline_cutoff,
        "features": FEATURES,
        "metrics_summary": {
            name: m["roc_auc"] for name, m in test_metrics.items()
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def main() -> None:
    bundle = train_and_save()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print("Dataset:", metrics["dataset"])
    print("\nHeld-out test metrics:")
    for name, m in metrics["models_test"].items():
        print(
            f"  {name:22s} AUC={m['roc_auc']:.3f}  Gini={m['gini']:.3f}  "
            f"KS={m['ks']:.3f}"
        )
    print("\n5-fold CV AUC (train):")
    for name, m in metrics["models_cv_train_auc"].items():
        print(f"  {name:22s} AUC={m['roc_auc']:.3f}")
    sc = metrics["scorecard"]
    print(
        f"\nBands: approve >= {sc['approve_cutoff']}, "
        f"review {sc['decline_cutoff']}–{sc['approve_cutoff']}, "
        f"decline < {sc['decline_cutoff']}"
    )
    print("\nTop IV features:")
    for row in sc["information_values"][:5]:
        print(f"  {row['feature']:26s} IV={row['iv']:.3f}")
    print(f"\nBundle saved to {MODEL_PATH} (bundle keys: {list(bundle)})")


if __name__ == "__main__":
    main()
