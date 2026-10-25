"""Dataset loading for the UCI Statlog German Credit dataset.

The dataset (1,000 loan applicants, 20 attributes, good/bad credit outcome)
is published by the UCI Machine Learning Repository:

    https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The raw file ``german.data`` is space-separated with coded categorical
values (``A11``, ``A34``, ...). The attribute names and code meanings come
from the companion file ``german.doc``. This module downloads the raw file
on first use into ``data/`` (git-ignored -- the dataset is *not* committed
to this repository) and returns a tidy DataFrame with clean column names.

Target convention used everywhere in this project:

    ``default`` = 1  ->  "bad" credit (class 2 in the raw file, defaulted)
    ``default`` = 0  ->  "good" credit (class 1 in the raw file)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "german.data"

# Column order of german.data, named after the attributes in german.doc.
COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate_pct",
    "personal_status",
    "other_debtors",
    "residence_since_years",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "dependents_count",
    "telephone",
    "foreign_worker",
    "class",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "personal_status",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "residence_since_years",
    "age_years",
    "existing_credits_count",
    "dependents_count",
]

FEATURES = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate_pct",
    "personal_status",
    "other_debtors",
    "residence_since_years",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "dependents_count",
    "telephone",
    "foreign_worker",
]

# Human-readable meanings of the codes, from german.doc. Used for reason
# codes (adverse-action explanations) in the scoring API.
CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "checking_status": {
        "A11": "checking account balance < 0",
        "A12": "checking account balance between 0 and 200",
        "A13": "checking account balance >= 200 / salary assigned",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all credits paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly so far",
        "A33": "delay in paying back loans in the past",
        "A34": "critical account / credits exist elsewhere",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture / equipment",
        "A43": "radio / television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other purpose",
    },
    "savings_status": {
        "A61": "savings < 100",
        "A62": "savings between 100 and 500",
        "A63": "savings between 500 and 1000",
        "A64": "savings >= 1000",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "employed < 1 year",
        "A73": "employed 1 to 4 years",
        "A74": "employed 4 to 7 years",
        "A75": "employed >= 7 years",
    },
    "personal_status": {
        "A91": "male, divorced / separated",
        "A92": "female, divorced / separated / married",
        "A93": "male, single",
        "A94": "male, married / widowed",
        "A95": "female, single",
    },
    "other_debtors": {
        "A101": "no other debtors / guarantors",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other property",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "installment plans with banks",
        "A142": "installment plans with stores",
        "A143": "no other installment plans",
    },
    "housing": {
        "A151": "rented housing",
        "A152": "own housing",
        "A153": "housing for free",
    },
    "job": {
        "A171": "unemployed / unskilled (non-resident)",
        "A172": "unskilled (resident)",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "no telephone registered",
        "A192": "telephone registered",
    },
    "foreign_worker": {
        "A201": "foreign worker",
        "A202": "not a foreign worker",
    },
}


def download_data(url: str = DATA_URL, dest: Path = DATA_PATH) -> Path:
    """Download the raw german.data file if it is not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed UCI URL
    return dest


def load_data(path: Path | None = None, download: bool = True) -> pd.DataFrame:
    """Load German Credit data with clean column names and a binary target.

    Returns a DataFrame with the 20 features in :data:`FEATURES` order plus
    a ``default`` column (1 = bad credit / defaulted, 0 = good credit).
    """
    if path is None:
        path = DATA_PATH
    if not Path(path).exists():
        if not download:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Call download_data() first "
                "or run `python -m src.train`."
            )
        download_data(dest=Path(path))

    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df["default"] = (df["class"] == 2).astype(int)
    df = df.drop(columns=["class"])
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col])
    return df[FEATURES + ["default"]]
