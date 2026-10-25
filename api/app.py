"""FastAPI scoring API for the credit-risk scorecard.

Endpoints
---------
GET  /health  -> service status and whether the scorecard bundle exists
POST /score   -> credit score, decision band, default probability and
                 adverse-action reason codes for one applicant

The request body for /score is a JSON object with the 20 German Credit
features (categorical values as their UCI codes, e.g. ``"A11"`` -- see
``data/README.md`` for the full code table). Example::

    {
      "checking_status": "A14", "duration_months": 24,
      "credit_history": "A32", "purpose": "A43",
      "credit_amount": 3500, "savings_status": "A61",
      "employment_since": "A73", "installment_rate_pct": 4,
      "personal_status": "A93", "other_debtors": "A101",
      "residence_since_years": 2, "property": "A123",
      "age_years": 35, "other_installment_plans": "A143",
      "housing": "A152", "existing_credits_count": 1,
      "job": "A173", "dependents_count": 1,
      "telephone": "A192", "foreign_worker": "A202"
    }

Model lifecycle: the bundle saved by ``python -m src.train``
(``models/scorecard.joblib``) is loaded lazily on the first /score call.
If it does not exist yet, the API trains the scorecard on the fly (the
dataset loader downloads the data if needed), so the service works from
a fresh clone:

    uvicorn api.app:app --reload     # from the repository root
"""

from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from src.data import FEATURES
from src.predict import MODEL_PATH, load_bundle, score_applicant

app = FastAPI(title="Credit Risk Scorecard API", version="1.0.0")

_bundle: dict | None = None


def get_bundle() -> dict:
    """Return the cached scorecard bundle, training on the fly if needed."""
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            from src.train import train_and_save

            train_and_save()
        _bundle = load_bundle()
    return _bundle


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": MODEL_PATH.exists() or _bundle is not None}


@app.post("/score")
def score(payload: dict = Body(...)) -> dict:
    missing = [c for c in FEATURES if c not in payload]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Missing applicant features: {missing}"
        )
    try:
        return score_applicant(payload, bundle=get_bundle())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
