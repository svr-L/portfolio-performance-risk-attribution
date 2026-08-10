"""Thin adapters for other portfolio engines.

The attribution engine intentionally does not generate scenarios.  It consumes
scenario return matrices produced upstream and applies the same probability-
weighted tail attribution used for historical returns.
"""
from __future__ import annotations

import pandas as pd

from .core import expected_shortfall_attribution


def scenario_expected_shortfall(
    scenario_returns: pd.DataFrame,
    weights,
    probabilities=None,
    *,
    alpha: float = 0.975,
    portfolio_value: float = 1.0,
):
    """Attribute scenario Expected Shortfall to assets.

    Parameters
    ----------
    scenario_returns:
        Rows are joint scenarios; columns are assets/instruments. Values are
        simple scenario returns over one common horizon.
    weights:
        Capital weights aligned to ``scenario_returns.columns`` and summing to 1.
    probabilities:
        Optional scenario probabilities. Equal probabilities are used when None.
    alpha:
        ES confidence level.
    portfolio_value:
        Currency value used to scale return losses into P&L losses.
    """
    if not isinstance(scenario_returns, pd.DataFrame):
        raise TypeError("scenario_returns must be a pandas DataFrame.")
    return expected_shortfall_attribution(
        scenario_returns,
        weights,
        alpha=alpha,
        probabilities=probabilities,
        portfolio_value=portfolio_value,
    )
