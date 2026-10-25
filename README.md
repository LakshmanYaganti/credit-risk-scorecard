# Credit Risk Scorecard

**Loan default prediction the way banks actually do it** — a Weight-of-Evidence
(WOE) / Information Value (IV) scorecard scaled to points, benchmarked against
plain logistic regression and XGBoost, validated with the metrics credit-risk
teams report (ROC-AUC, Gini, KS, calibration), and served behind a FastAPI
scoring endpoint that returns **adverse-action reason codes** with every score.

Built on the UCI Statlog German Credit dataset: 1,000 loan applicants,
20 features, 30% default rate.

## Why scorecards still run consumer lending

Gradient boosting wins Kaggle leaderboards, but application scorecards —
WOE binning + logistic regression, scaled to points — still underwrite a
huge share of the world's credit cards, auto loans, and personal loans.
The reasons are regulatory and operational, not nostalgic:

- **Interpretability.** Every point of an applicant's score traces to a
  specific feature value ("checking account overdrawn: −43 points vs the
  best bin"). Model risk teams (and SR 11-7-style validation) can audit
  each coefficient's direction and magnitude.
- **Adverse action.** When an applicant is declined, US ECOA / Regulation B
  requires the lender to state specific reasons. A points scorecard
  generates those reason codes for free — the features where the applicant
  lost the most points. A black-box model needs a post-hoc explainer bolted
  on; the scorecard *is* the explanation.
- **Stability & monitoring.** Coarse WOE bins are robust to drift and
  outliers, scores map monotonically to default odds, and the whole model
  fits on one page for a credit committee.

This project shows the full workflow — and honestly measures what (if
anything) the interpretable model gives up against XGBoost.

## Method

1. **Binning + WOE.** Categoricals keep one bin per category; numerics get
   quantile bins (≤10). Each bin's WOE = ln(share of good / share of bad),
   with 0.5 smoothing.
2. **Information Value.** IV = Σ (share_good − share_bad) × WOE ranks the
   features before any model is fit.
3. **Logistic regression on WOE features**, then **points scaling**:
   score = offset + factor × ln(odds of good), with base score **600** at
   odds **50 : 1** and **PDO 20** (doubling the good:bad odds adds exactly
   20 points). Per-feature points sum exactly to the applicant's score.
4. **Decision bands on modelled PD** — approve if PD ≤ 15%
   (score ≥ **537**), decline if PD ≥ 50% (score < **487**), manual review
   in between — converted to score cutoffs through the same scaling.
5. **Validation** on a stratified 70/30 split (seed 42), plus 5-fold CV
   AUCs, since 1,000 rows is small.

## Results (held-out test set, n = 300, 90 defaults)

| Model | ROC-AUC | Gini | KS | 5-fold CV AUC (train) |
|---|---|---|---|---|
| **Scorecard (WOE + LR)** | **0.800** | **0.600** | **0.494** | 0.762 |
| Logistic regression (one-hot) | 0.801 | 0.602 | 0.494 | 0.763 |
| XGBoost | 0.796 | 0.592 | 0.459 | 0.764 |

The interpretable scorecard **ties** both benchmarks on ranking metrics.
On small tabular lending data this is the usual finding — and the
scorecard additionally ships with calibration, points, and reason codes.

**Top features by Information Value** (train):

| Feature | IV | Reading |
|---|---|---|
| checking_status | 0.630 | strong — overdrawn (A11) defaults at 49%, no account (A14) at 12% |
| duration_months | 0.258 | medium — longer terms are riskier |
| credit_history | 0.252 | medium |
| savings_status | 0.221 | medium |
| property | 0.139 | medium |
| employment_since | 0.139 | medium |

At the other end, `dependents_count`, `residence_since_years`, and
`existing_credits_count` have IV ≈ 0.00 — the scorecard effectively
ignores them, which the points table makes visible.

**Decision bands realized on the test set:**

| Band | Policy | Share of applicants | Bad rate |
|---|---|---|---|
| Approve | score ≥ 537 (PD ≤ 15%) | 36.3% | **9.2%** |
| Review | 487 ≤ score < 537 | 42.7% | 29.7% |
| Decline | score < 487 (PD ≥ 50%) | 21.0% | **66.7%** |

At the approve cutoff, 89% of all defaults are kept out of the approved
book (80 of 90), at the price of sending 111 of 210 good applicants to
review/decline — the classic approval-rate vs bad-rate tradeoff, tabulated
for ten cutoffs in `reports/metrics.json` (e.g. a stricter cutoff of 595
approves just 5% of applicants at a 0% observed bad rate).

**Calibration:** predicted PD tracks observed default rates closely across
deciles (e.g. mean predicted 0.36 vs observed 0.37; 0.59 vs 0.67 in the
riskiest bands) — see the notebook for the full curve.

**Reason codes in action** (notebook demo applicant): score **461 →
DECLINE** (PD 71%), top reasons: *loan duration 36 months; checking
account balance < 0; foreign worker; employed < 1 year*.

## Fair-lending note

This dataset dates from 1970s–90s Germany and includes attributes that are
protected classes today — sex/marital status (`personal_status`),
`foreign_worker`, and `age_years`. They are retained here to reproduce the
classic benchmark faithfully and to keep the IV/points audit honest (you
can see exactly how much each contributes). A production scorecard under
ECOA-style rules would exclude them, test the performance delta, and
document the decision. Treat this repo as a methods demonstration, not a
deployable underwriting policy.

## Project structure

```
credit-risk-scorecard/
├── api/app.py                  # FastAPI: GET /health, POST /score
├── data/README.md              # dataset provenance + full code table
├── notebooks/
│   └── 01_credit_risk_modeling.ipynb   # executed EDA -> IV -> models -> cutoffs
├── reports/metrics.json        # real metrics from the training run
├── src/
│   ├── data.py                 # UCI download/load, clean names, code labels
│   ├── scorecard.py            # WOE binning, IV, points-scaled scorecard
│   ├── train.py                # scorecard + LogReg + XGBoost, bands, metrics
│   ├── evaluate.py             # AUC, Gini, KS, calibration, cutoff tradeoff
│   └── predict.py              # score one applicant + reason codes
└── tests/test_pipeline.py      # WOE/IV math, scaling, bands, API tests
```

## Quickstart

```bash
pip install -r requirements.txt

python -m src.train          # downloads data, trains, writes reports/metrics.json
                             # and models/scorecard.joblib (git-ignored)

uvicorn api.app:app --reload # scoring API on http://localhost:8000
                             # (trains on first /score call if no bundle exists)

pytest                       # 11 tests
```

Score an applicant:

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "checking_status": "A14", "duration_months": 24, "credit_history": "A32",
  "purpose": "A43", "credit_amount": 3500, "savings_status": "A61",
  "employment_since": "A73", "installment_rate_pct": 4,
  "personal_status": "A93", "other_debtors": "A101",
  "residence_since_years": 2, "property": "A123", "age_years": 35,
  "other_installment_plans": "A143", "housing": "A152",
  "existing_credits_count": 1, "job": "A173", "dependents_count": 1,
  "telephone": "A192", "foreign_worker": "A202"}'
```

```json
{"credit_score": 585, "band": "approve", "default_probability": 0.0325,
 "reason_codes": ["Loan duration: 24", "Savings: savings < 100",
                  "Installment burden: 4",
                  "Credit history: existing credits paid back duly so far"],
 "points_breakdown": {"checking_status": 51.8, "...": "..."}}
```

Categorical values are the raw UCI codes — see `data/README.md` for the
full table.

## Tech stack

Python · pandas · scikit-learn · XGBoost · FastAPI · matplotlib/seaborn ·
pytest · Jupyter

## Dataset & license

Data: UCI Statlog German Credit (Hofmann), via the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) —
downloaded at runtime, not redistributed here. Code: MIT (see `LICENSE`).

## Author

**Lakshman Sai Yaganti** — Data Scientist @ Square · fraud detection &
credit risk · [github.com/LakshmanYaganti](https://github.com/LakshmanYaganti)
