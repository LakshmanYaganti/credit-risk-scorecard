"""Score a single applicant with the trained scorecard bundle.

Returns the scaled credit score, the decision band, the modelled default
probability, and **reason codes** -- the per-feature points the applicant
lost versus the best possible bin of each feature, phrased in plain
language. Reason codes are how lenders satisfy adverse-action notice
requirements (e.g. ECOA in the US): a declined applicant is entitled to
the specific reasons, and a points scorecard produces them naturally.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import CATEGORY_LABELS, CATEGORICAL_FEATURES, FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "scorecard.joblib"

FEATURE_DISPLAY = {
    "checking_status": "Checking account status",
    "duration_months": "Loan duration",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount",
    "savings_status": "Savings",
    "employment_since": "Employment stability",
    "installment_rate_pct": "Installment burden",
    "personal_status": "Personal status",
    "other_debtors": "Other debtors / guarantors",
    "residence_since_years": "Time at current residence",
    "property": "Property / collateral",
    "age_years": "Age",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits_count": "Number of existing credits",
    "job": "Job type",
    "dependents_count": "Number of dependents",
    "telephone": "Telephone on file",
    "foreign_worker": "Foreign worker status",
}


def load_bundle(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def _reason_text(feature: str, applicant: dict) -> str:
    display = FEATURE_DISPLAY.get(feature, feature)
    if feature in CATEGORICAL_FEATURES:
        label = CATEGORY_LABELS.get(feature, {}).get(
            str(applicant[feature]), str(applicant[feature])
        )
        return f"{display}: {label}"
    return f"{display}: {applicant[feature]}"


def score_applicant(applicant: dict, bundle: dict | None = None) -> dict:
    """Score one applicant dict keyed by the German Credit feature names.

    Categorical values use the raw UCI codes (``"A11"``, ``"A34"``, ...)
    documented in ``data/README.md``; numeric values are plain numbers.
    """
    missing = [f for f in FEATURES if f not in applicant]
    if missing:
        raise ValueError(f"Missing applicant features: {missing}")
    if bundle is None:
        bundle = load_bundle()
    card = bundle["scorecard"]

    frame = pd.DataFrame([{f: applicant[f] for f in FEATURES}])
    score = float(card.score(frame)[0])
    default_proba = float(card.predict_default_proba(frame)[0])

    if score >= bundle["approve_cutoff"]:
        band = "approve"
    elif score >= bundle["decline_cutoff"]:
        band = "review"
    else:
        band = "decline"

    # Reason codes: features where the applicant loses the most points
    # relative to that feature's best bin.
    breakdown = card.points_breakdown(applicant)
    losses = []
    for feature in FEATURES:
        best_points = max(card.points_table[feature].values())
        lost = best_points - breakdown[feature]
        if lost > 0:
            losses.append((feature, lost))
    losses.sort(key=lambda item: item[1], reverse=True)
    reason_codes = [
        _reason_text(feature, applicant) for feature, _lost in losses[:4]
    ]

    return {
        "credit_score": round(score),
        "band": band,
        "default_probability": round(default_proba, 4),
        "reason_codes": reason_codes,
        "points_breakdown": {k: round(v, 1) for k, v in breakdown.items()},
    }
