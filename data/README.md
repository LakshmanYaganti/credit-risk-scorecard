# Dataset: UCI Statlog German Credit

- **Source:** UCI Machine Learning Repository —
  [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
  donated by Prof. Dr. Hans Hofmann, University of Hamburg.
- **Raw file:** `german.data` from
  `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data`
  (attribute documentation in the companion `german.doc`).
- **Size:** 1,000 loan applicants × 20 attributes + class label.
  700 good / 300 bad (30% default rate).
- **License/terms:** free for research use via the UCI repository; cite
  the UCI ML Repository if you reuse it.

The dataset is **not committed** to this repository. `src/data.py`
downloads it into this folder on first use (or run `python -m src.data`
indirectly via `python -m src.train`).

## Target convention

The raw class label is `1 = good`, `2 = bad`. This project converts it to
`default`: **1 = bad (defaulted), 0 = good**.

## Features and codes

Numeric features: `duration_months`, `credit_amount`,
`installment_rate_pct` (installment as % of disposable income),
`residence_since_years`, `age_years`, `existing_credits_count`,
`dependents_count`.

Categorical features use the raw UCI codes in API requests:

| Feature | Codes |
|---|---|
| `checking_status` | A11 <0 · A12 0–200 · A13 ≥200 / salary assigned · A14 no checking account |
| `credit_history` | A30 no credits / all paid · A31 all paid at this bank · A32 paid so far · A33 past payment delay · A34 critical account |
| `purpose` | A40 car (new) · A41 car (used) · A42 furniture/equipment · A43 radio/TV · A44 domestic appliances · A45 repairs · A46 education · A47 vacation · A48 retraining · A49 business · A410 other |
| `savings_status` | A61 <100 · A62 100–500 · A63 500–1000 · A64 ≥1000 · A65 unknown / none |
| `employment_since` | A71 unemployed · A72 <1 yr · A73 1–4 yrs · A74 4–7 yrs · A75 ≥7 yrs |
| `personal_status` | A91 male div/sep · A92 female div/sep/married · A93 male single · A94 male married/widowed · A95 female single |
| `other_debtors` | A101 none · A102 co-applicant · A103 guarantor |
| `property` | A121 real estate · A122 building-society savings / life insurance · A123 car or other · A124 unknown / none |
| `other_installment_plans` | A141 bank · A142 stores · A143 none |
| `housing` | A151 rent · A152 own · A153 for free |
| `job` | A171 unemployed/unskilled (non-resident) · A172 unskilled (resident) · A173 skilled employee/official · A174 management / self-employed / highly qualified |
| `telephone` | A191 none · A192 registered |
| `foreign_worker` | A201 yes · A202 no |

Amounts are in Deutsch Mark (DM), the dataset's original currency.
