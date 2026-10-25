"""Tests for the credit-risk scorecard pipeline.

Coverage:
- WOE / IV math on a tiny hand-checkable frame
- IV ranking on the real dataset (checking_status must dominate)
- scorecard points scaling (base score at base odds, PDO property)
- points-breakdown consistency (per-feature points sum to the score)
- decision bands on extreme real applicants
- FastAPI /health and /score behaviour
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_data
from src.scorecard import Scorecard, information_value, woe_iv_table


# ----------------------------------------------------------------------
# WOE / IV math
# ----------------------------------------------------------------------
def test_woe_iv_perfect_separation():
    values = pd.Series(["a", "a", "b", "b"])
    target = pd.Series([0, 0, 1, 1])  # 1 = bad
    table = woe_iv_table(values, target)
    # With 0.5 smoothing: WOE_a = ln((2.5/3) / (0.5/3)) = ln(5)
    assert table.loc["a", "woe"] == pytest.approx(np.log(5), abs=1e-6)
    assert table.loc["b", "woe"] == pytest.approx(-np.log(5), abs=1e-6)
    iv = information_value(values, target)
    assert iv == pytest.approx(2 * (2.0 / 3.0) * np.log(5), abs=1e-6)


def test_woe_iv_no_signal_is_zero():
    values = pd.Series(["a", "b", "a", "b"])
    target = pd.Series([0, 0, 1, 1])
    assert information_value(values, target) == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# Real-data fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return load_data()


@pytest.fixture(scope="module")
def fitted(data):
    train_df, test_df = train_test_split(
        data, test_size=0.30, random_state=42, stratify=data["default"]
    )
    card = Scorecard(CATEGORICAL_FEATURES, NUMERIC_FEATURES).fit(train_df)
    return card, train_df, test_df


def test_iv_ranking_checking_status_first(fitted):
    card, _train, _test = fitted
    top = card.iv_table.iloc[0]
    assert top["feature"] == "checking_status"
    assert top["iv"] > 0.3  # "strong" by the usual IV rules of thumb


def test_points_scaling_base_and_pdo(fitted):
    card, _train, _test = fitted
    # Base: odds of good = base_odds (50:1) must map to base_score (600).
    score_at_base = card.offset + card.factor * np.log(card.base_odds)
    assert score_at_base == pytest.approx(card.base_score)
    # PDO: doubling the odds adds exactly PDO (20) points.
    for odds in (1.0, 7.0, 50.0):
        s1 = card.offset + card.factor * np.log(odds)
        s2 = card.offset + card.factor * np.log(2 * odds)
        assert s2 - s1 == pytest.approx(card.pdo)


def test_points_breakdown_sums_to_score(fitted):
    card, _train, test_df = fitted
    row = test_df.iloc[0]
    applicant = {f: row[f] for f in FEATURES}
    breakdown = card.points_breakdown(applicant)
    direct = float(card.score(test_df.iloc[[0]])[0])
    assert sum(breakdown.values()) == pytest.approx(direct, abs=0.5)
    assert set(breakdown) == set(FEATURES)


def test_score_monotone_in_probability(fitted):
    card, _train, test_df = fitted
    scores = card.score(test_df)
    probas = card.predict_default_proba(test_df)
    # Higher score must always mean lower modelled default probability.
    order = np.argsort(scores)
    assert np.all(np.diff(probas[order]) <= 1e-9)


# ----------------------------------------------------------------------
# Prediction / bands
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle(fitted):
    card, _train, _test = fitted
    from src.train import derive_bands

    approve, decline = derive_bands(card)
    return {"scorecard": card, "approve_cutoff": approve, "decline_cutoff": decline}


def test_extreme_applicants_get_extreme_bands(bundle, fitted):
    from src.predict import score_applicant

    card, _train, test_df = fitted
    scores = card.score(test_df)
    worst = test_df.iloc[int(np.argmin(scores))]
    best = test_df.iloc[int(np.argmax(scores))]
    worst_result = score_applicant({f: worst[f] for f in FEATURES}, bundle=bundle)
    best_result = score_applicant({f: best[f] for f in FEATURES}, bundle=bundle)
    assert worst_result["band"] == "decline"
    assert best_result["band"] == "approve"
    assert best_result["credit_score"] > worst_result["credit_score"]
    assert worst_result["default_probability"] > best_result["default_probability"]
    assert len(worst_result["reason_codes"]) >= 1


def test_score_applicant_missing_feature_raises(bundle):
    from src.predict import score_applicant

    with pytest.raises(ValueError, match="Missing applicant features"):
        score_applicant({"checking_status": "A14"}, bundle=bundle)


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
APPLICANT = {
    "checking_status": "A14",
    "duration_months": 24,
    "credit_history": "A32",
    "purpose": "A43",
    "credit_amount": 3500,
    "savings_status": "A61",
    "employment_since": "A73",
    "installment_rate_pct": 4,
    "personal_status": "A93",
    "other_debtors": "A101",
    "residence_since_years": 2,
    "property": "A123",
    "age_years": 35,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits_count": 1,
    "job": "A173",
    "dependents_count": 1,
    "telephone": "A192",
    "foreign_worker": "A202",
}


@pytest.fixture(scope="module")
def client(bundle):
    from fastapi.testclient import TestClient

    import api.app as app_module

    app_module._bundle = bundle  # inject the fitted bundle; no retraining
    return TestClient(app_module.app)


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_score(client):
    response = client.post("/score", json=APPLICANT)
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"credit_score", "band", "default_probability", "reason_codes"}
    assert body["band"] in {"approve", "review", "decline"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert len(body["reason_codes"]) >= 1


def test_api_score_missing_features_422(client):
    response = client.post("/score", json={"checking_status": "A14"})
    assert response.status_code == 422
