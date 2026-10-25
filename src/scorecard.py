"""WOE binning, Information Value, and a points-based credit scorecard.

This is the classic scorecard pipeline still used across consumer lending:

1. **Binning + Weight of Evidence (WOE).** Every feature is split into
   bins (one bin per category for categorical features, quantile bins for
   numeric ones). For each bin::

       WOE = ln( share of good applicants in bin / share of bad in bin )

   A positive WOE means the bin leans "good", negative means "bad".

2. **Information Value (IV).** Per feature::

       IV = sum_bins (share_good - share_bad) * WOE

   IV ranks predictive strength (rough guide: <0.02 useless, 0.02-0.1 weak,
   0.1-0.3 medium, >0.3 strong).

3. **Logistic regression on WOE features.** Monotonic, interpretable, and
   the coefficients translate directly into scorecard points.

4. **Points scaling.** Scores follow the industry convention::

       score = offset + factor * ln(odds of good)
       factor = PDO / ln(2)          (points to double the odds)
       offset = base_score - factor * ln(base_odds)

   With base score 600 at odds 50:1 and PDO = 20, an applicant whose
   good:bad odds double earns exactly 20 more points.

Each feature/bin gets an integer-friendly points value, so a single
applicant's score decomposes into per-feature contributions -- which is
exactly what adverse-action reason codes need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SMOOTHING = 0.5  # added to good/bad counts per bin to avoid log(0)


# ----------------------------------------------------------------------
# WOE / IV primitives
# ----------------------------------------------------------------------
def woe_iv_table(
    values: pd.Series, target: pd.Series, bins: pd.Series | None = None
) -> pd.DataFrame:
    """Return per-bin good/bad counts, WOE and IV contribution.

    ``values`` is the raw feature, ``target`` is 1 = bad (default),
    0 = good. ``bins`` optionally assigns each row to a bin label; when
    omitted, each distinct value of ``values`` is its own bin.
    """
    bin_labels = values if bins is None else bins
    frame = pd.DataFrame({"bin": bin_labels, "target": target})
    grouped = frame.groupby("bin", observed=True)["target"]
    table = pd.DataFrame(
        {
            "count": grouped.size(),
            "bad": grouped.sum(),
        }
    )
    table["good"] = table["count"] - table["bad"]

    total_good = table["good"].sum()
    total_bad = table["bad"].sum()
    good_dist = (table["good"] + SMOOTHING) / (total_good + SMOOTHING * len(table))
    bad_dist = (table["bad"] + SMOOTHING) / (total_bad + SMOOTHING * len(table))
    table["woe"] = np.log(good_dist / bad_dist)
    table["iv_contrib"] = (good_dist - bad_dist) * table["woe"]
    return table


def information_value(values: pd.Series, target: pd.Series, bins=None) -> float:
    """Total Information Value of a feature against the target."""
    return float(woe_iv_table(values, target, bins)["iv_contrib"].sum())


def quantile_bins(values: pd.Series, max_bins: int = 10) -> tuple[pd.Series, np.ndarray]:
    """Assign numeric values to quantile bins; return (bin ids, edges)."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, max_bins + 1)))
    if len(edges) <= 2:  # (near-)constant feature: single bin
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index), edges
    bin_ids = np.searchsorted(edges[1:-1], values, side="right")
    return (
        pd.Series(bin_ids, index=values.index),
        edges,
    )


def bin_index_for_value(edges: np.ndarray, value: float) -> int:
    """Bin index for a single numeric value given stored quantile edges."""
    if len(edges) <= 2:
        return 0
    return int(np.searchsorted(edges[1:-1], value, side="right"))


# ----------------------------------------------------------------------
# Scorecard
# ----------------------------------------------------------------------
class Scorecard:
    """Logistic-regression scorecard on WOE-transformed features.

    Parameters follow the standard scaling: an applicant with good:bad
    odds of ``base_odds`` receives ``base_score`` points, and every
    doubling of the odds (``pdo``) adds ``pdo`` points.
    """

    def __init__(
        self,
        categorical_features: list[str],
        numeric_features: list[str],
        base_score: float = 600.0,
        base_odds: float = 50.0,
        pdo: float = 20.0,
        max_numeric_bins: int = 10,
        random_state: int = 42,
    ) -> None:
        self.categorical_features = list(categorical_features)
        self.numeric_features = list(numeric_features)
        self.features = self.categorical_features + self.numeric_features
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.max_numeric_bins = max_numeric_bins
        self.random_state = random_state

        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

        # Fitted state
        self.woe_maps: dict[str, dict] = {}
        self.numeric_edges: dict[str, np.ndarray] = {}
        self.iv_table: pd.DataFrame | None = None
        self.model: LogisticRegression | None = None
        self.points_table: dict[str, dict] = {}

    # -- internal helpers ------------------------------------------------
    def _bin_series(self, df: pd.DataFrame, feature: str, fit: bool) -> pd.Series:
        if feature in self.categorical_features:
            return df[feature].astype(str)
        if fit:
            bins, edges = quantile_bins(df[feature], self.max_numeric_bins)
            self.numeric_edges[feature] = edges
            return bins
        edges = self.numeric_edges[feature]
        return df[feature].apply(lambda v: bin_index_for_value(edges, v))

    def _woe_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for feature in self.features:
            bins = self._bin_series(df, feature, fit=False)
            woe_map = self.woe_maps[feature]
            if feature in self.categorical_features:
                out[feature] = bins.map(woe_map).fillna(0.0)
            else:
                out[feature] = bins.map(lambda b: woe_map.get(int(b), 0.0))
        return pd.DataFrame(out, index=df.index)[self.features]

    # -- fitting ----------------------------------------------------------
    def fit(self, df: pd.DataFrame, target_col: str = "default") -> "Scorecard":
        target = df[target_col]
        iv_rows = {}
        for feature in self.features:
            bins = self._bin_series(df, feature, fit=True)
            table = woe_iv_table(df[feature], target, bins)
            if feature in self.categorical_features:
                self.woe_maps[feature] = {
                    str(k): float(v) for k, v in table["woe"].items()
                }
            else:
                self.woe_maps[feature] = {
                    int(k): float(v) for k, v in table["woe"].items()
                }
            iv_rows[feature] = float(table["iv_contrib"].sum())
        self.iv_table = (
            pd.DataFrame({"feature": list(iv_rows), "iv": list(iv_rows.values())})
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

        x_woe = self._woe_transform(df)
        self.model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.model.fit(x_woe, target)
        self._build_points_table()
        return self

    def _build_points_table(self) -> None:
        """Allocate scaled points per feature/bin.

        The LR models P(bad), so ln(odds of good) = -(intercept + sum
        coef_j * WOE_j). Allocating the negated terms evenly::

            points_j(bin) = factor * (-coef_j * WOE_j(bin) - intercept / n)
                            + offset / n

        so that sum_j points_j == offset + factor * ln(odds of good).
        """
        assert self.model is not None
        n = len(self.features)
        coefs = dict(zip(self.features, self.model.coef_[0]))
        intercept_share = float(self.model.intercept_[0]) / n
        offset_share = self.offset / n
        for feature in self.features:
            self.points_table[feature] = {
                bin_key: self.factor * (-coefs[feature] * woe - intercept_share)
                + offset_share
                for bin_key, woe in self.woe_maps[feature].items()
            }

    # -- scoring ----------------------------------------------------------
    def log_odds_good(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        x_woe = self._woe_transform(df)
        # decision_function gives log-odds of the positive class (bad);
        # negate for log-odds of good.
        return -self.model.decision_function(x_woe)

    def predict_default_proba(self, df: pd.DataFrame) -> np.ndarray:
        log_odds = self.log_odds_good(df)
        odds_good = np.exp(log_odds)
        return 1.0 / (1.0 + odds_good)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Scaled credit score (higher = safer applicant)."""
        return self.offset + self.factor * self.log_odds_good(df)

    def points_breakdown(self, applicant: dict) -> dict[str, float]:
        """Per-feature points for one applicant (sums to the total score)."""
        breakdown = {}
        for feature in self.features:
            value = applicant[feature]
            if feature in self.categorical_features:
                bin_key = str(value)
            else:
                bin_key = bin_index_for_value(self.numeric_edges[feature], float(value))
            breakdown[feature] = float(self.points_table[feature].get(bin_key, 0.0))
        return breakdown

    def score_one(self, applicant: dict) -> float:
        return float(sum(self.points_breakdown(applicant).values()))
