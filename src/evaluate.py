"""Evaluation metrics the way credit-risk teams actually report them.

Beyond plain accuracy (meaningless for lending decisions), this module
computes:

- **ROC-AUC** -- overall ranking quality.
- **Gini coefficient** -- ``2 * AUC - 1``, the industry-standard summary.
- **KS statistic** -- the maximum separation between the cumulative
  distributions of good and bad applicants over the score; the headline
  metric on most scorecard validation reports.
- **Calibration data** -- mean predicted default probability vs the
  observed default rate in probability bands. A scorecard can rank well
  and still be badly calibrated, which breaks pricing and provisioning.
- **Cutoff tradeoff table** -- for each candidate score cutoff: approval
  rate, bad rate among approved applicants, and share of all bads
  rejected. This is the table a credit committee actually argues over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def roc_auc(y_true: np.ndarray, default_proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, default_proba))


def gini(y_true: np.ndarray, default_proba: np.ndarray) -> float:
    return 2.0 * roc_auc(y_true, default_proba) - 1.0


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation of good/bad score distributions.

    ``score`` is a credit score (higher = safer). KS is the largest gap
    between the cumulative share of bads and the cumulative share of
    goods when applicants are ordered from riskiest to safest.
    """
    order = np.argsort(score)  # riskiest first
    y_sorted = np.asarray(y_true)[order]
    total_bad = y_sorted.sum()
    total_good = len(y_sorted) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0
    cum_bad = np.cumsum(y_sorted) / total_bad
    cum_good = np.cumsum(1 - y_sorted) / total_good
    return float(np.max(np.abs(cum_bad - cum_good)))


def roc_data(y_true: np.ndarray, default_proba: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, default_proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def calibration_data(
    y_true: np.ndarray, default_proba: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Observed default rate vs mean predicted PD per probability band."""
    frame = pd.DataFrame({"y": y_true, "p": default_proba})
    frame["band"] = pd.qcut(frame["p"], n_bins, duplicates="drop")
    grouped = frame.groupby("band", observed=True).agg(
        count=("y", "size"),
        mean_predicted_pd=("p", "mean"),
        observed_default_rate=("y", "mean"),
    )
    return grouped.reset_index(drop=True).to_dict(orient="records")


def confusion_at_cutoff(
    y_true: np.ndarray, score: np.ndarray, cutoff: float
) -> dict:
    """Confusion matrix when applicants scoring >= cutoff are approved.

    A "positive" prediction here means *predicted bad* (declined), which
    matches how the raw sklearn matrix is laid out with labels [0, 1].
    """
    predicted_bad = (score < cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_bad, labels=[0, 1]).ravel()
    return {
        "cutoff": float(cutoff),
        "true_good_approved": int(tn),
        "true_good_declined": int(fp),
        "true_bad_approved": int(fn),
        "true_bad_declined": int(tp),
    }


def cutoff_tradeoff(
    y_true: np.ndarray, score: np.ndarray, cutoffs: list[float]
) -> pd.DataFrame:
    """Approval rate vs bad rate for a range of score cutoffs."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    total = len(y_true)
    total_bad = y_true.sum()
    rows = []
    for cutoff in cutoffs:
        approved = score >= cutoff
        n_approved = int(approved.sum())
        bads_approved = int(y_true[approved].sum())
        rows.append(
            {
                "cutoff": float(cutoff),
                "approval_rate": n_approved / total if total else 0.0,
                "bad_rate_among_approved": (
                    bads_approved / n_approved if n_approved else 0.0
                ),
                "share_of_bads_rejected": (
                    1.0 - bads_approved / total_bad if total_bad else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def model_metrics(y_true: np.ndarray, default_proba: np.ndarray) -> dict:
    """AUC / Gini / KS bundle for one model.

    KS is computed on the implied score (negative probability), which is
    a monotone transform of the probability, so KS is identical either
    way.
    """
    return {
        "roc_auc": roc_auc(y_true, default_proba),
        "gini": gini(y_true, default_proba),
        "ks": ks_statistic(y_true, -np.asarray(default_proba)),
    }
