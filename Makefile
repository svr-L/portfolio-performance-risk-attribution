"""Release-oriented validation and reconciliation gate for public v1."""
from __future__ import annotations

from .core import (
    PortfolioAttributionEngine,
    brinson_multi_period,
    brinson_multi_period_reference,
    compound_returns,
    period_start_weights,
)

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


def validate_inputs(
    returns: pd.DataFrame,
    portfolio_weights,
    benchmark_weights=None,
    classifications: pd.DataFrame | None = None,
    factor_returns: pd.DataFrame | None = None,
) -> list[ValidationIssue]:
    """Return all obvious contract violations instead of failing at the first."""
    issues: list[ValidationIssue] = []
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        return [ValidationIssue("returns", "must be a non-empty DataFrame")]
    if not isinstance(returns.index, pd.DatetimeIndex):
        issues.append(ValidationIssue("returns", "index must be a DatetimeIndex"))
    elif not returns.index.is_monotonic_increasing:
        issues.append(ValidationIssue("returns", "index must be sorted ascending"))
    if returns.columns.duplicated().any():
        issues.append(ValidationIssue("returns", "duplicate asset columns found"))
    numeric = returns.apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy()).any():
        issues.append(ValidationIssue("returns", "contains +/-inf"))
    if (numeric <= -1.0).any().any():
        issues.append(ValidationIssue("returns", "contains observations <= -100%, incompatible with geometric linking"))

    assets = returns.columns

    def check_weights(obj, name):
        if obj is None:
            return
        if isinstance(obj, pd.DataFrame):
            missing = assets.difference(obj.columns)
            if len(missing):
                issues.append(ValidationIssue(name, f"missing assets {missing.tolist()}"))
                return
            w = obj.reindex(columns=assets)
            if w.isna().any().any():
                issues.append(ValidationIssue(name, "contains NaN"))
            if not np.allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-8):
                issues.append(ValidationIssue(name, "each row must sum to one"))
        else:
            s = pd.Series(obj, index=assets if not isinstance(obj, pd.Series) else None, dtype=float)
            if isinstance(obj, pd.Series):
                s = obj.reindex(assets).astype(float)
            if s.isna().any():
                issues.append(ValidationIssue(name, "does not cover every asset"))
            elif not np.isclose(float(s.sum()), 1.0, atol=1e-8):
                issues.append(ValidationIssue(name, f"must sum to one; got {float(s.sum()):.12f}"))

    check_weights(portfolio_weights, "portfolio_weights")
    check_weights(benchmark_weights, "benchmark_weights")

    if classifications is not None:
        if not isinstance(classifications, pd.DataFrame) or classifications.empty:
            issues.append(ValidationIssue("classifications", "must be a non-empty DataFrame"))
        else:
            miss = assets.difference(classifications.index)
            if len(miss):
                issues.append(ValidationIssue("classifications", f"missing assets {miss.tolist()}"))
            if classifications.reindex(assets).isna().any().any():
                issues.append(ValidationIssue("classifications", "contains missing classifications"))

    if factor_returns is not None:
        if not isinstance(factor_returns, pd.DataFrame) or factor_returns.empty:
            issues.append(ValidationIssue("factor_returns", "must be a non-empty DataFrame"))
        elif not isinstance(factor_returns.index, pd.DatetimeIndex):
            issues.append(ValidationIssue("factor_returns", "index must be a DatetimeIndex"))
        elif factor_returns.columns.duplicated().any():
            issues.append(ValidationIssue("factor_returns", "duplicate factor columns found"))

    return issues


def release_gate(
    returns: pd.DataFrame,
    portfolio_weights,
    benchmark_weights=None,
    classifications: pd.DataFrame | None = None,
    factor_returns: pd.DataFrame | None = None,
    *,
    identity_tol: float = 1e-10,
    meaning_tol: float = 1e-10,
) -> tuple[pd.DataFrame, bool]:
    """Run the stable public-v1 blocks and fail on reconciliation/meaning errors."""
    rows: list[dict] = []

    def add(block: str, kind: str, value: float, tol: float):
        value = float(value)
        rows.append({
            "block": block,
            "kind": kind,
            "value": value,
            "tolerance": tol,
            "passed": bool(np.isfinite(value) and abs(value) <= tol),
        })

    issues = validate_inputs(
        returns, portfolio_weights, benchmark_weights, classifications, factor_returns
    )
    rows.append({
        "block": "input_contracts",
        "kind": "meaning",
        "value": float(len(issues)),
        "tolerance": 0.0,
        "passed": not issues,
    })
    if issues:
        return pd.DataFrame(rows).set_index(["block", "kind"]), False

    eng = PortfolioAttributionEngine(
        returns=returns,
        portfolio_weights=portfolio_weights,
        benchmark_weights=benchmark_weights,
        classifications=classifications,
        factor_returns=factor_returns,
    )

    _, ck = eng.linked_asset_performance(basis="native")
    add("linked_performance", "identity", ck["linking_error"], identity_tol)
    realised = float((1.0 + eng.portfolio_returns).prod() - 1.0)
    add(
        "linked_performance",
        "meaning",
        ck["cumulative_portfolio_return"] - realised,
        meaning_tol,
    )

    _, vck = eng.volatility()
    add("volatility", "identity", vck["volatility_additivity_error"], identity_tol)

    _, esck, _ = eng.expected_shortfall(alpha=0.975)
    denom = max(abs(float(esck["Expected_Shortfall"])), 1e-14)
    add("expected_shortfall", "identity", esck["additivity_error"] / denom, identity_tol)

    _, ddck = eng.max_drawdown()
    add("max_drawdown", "identity", ddck["additivity_error"], identity_tol)

    if benchmark_weights is not None:
        _, teck = eng.tracking_error()
        add("tracking_error", "identity", teck["tracking_error_additivity_error"], identity_tol)

    if benchmark_weights is not None and classifications is not None:
        level = classifications.columns[0]

        # Exact native-frequency Brinson must reconcile both algebraically and
        # economically to independently calculated realised returns.
        _, _, bck = eng.brinson(level=level, method="BF", basis="native")
        add("brinson_native", "identity", bck["linking_error"], identity_tol)

        realised_rp = float((1.0 + eng.portfolio_returns).prod() - 1.0)
        realised_rb = float((1.0 + eng.benchmark_returns).prod() - 1.0)
        add(
            "brinson_native_portfolio",
            "meaning",
            bck["cumulative_portfolio_return"] - realised_rp,
            meaning_tol,
        )
        add(
            "brinson_native_benchmark",
            "meaning",
            bck["cumulative_benchmark_return"] - realised_rb,
            meaning_tol,
        )

        # The coarser period view is allowed to differ, but only if it explicitly
        # identifies itself as an approximation. The gap is surfaced as a
        # diagnostic rather than hidden behind a successful reconciliation.
        _, _, pck = eng.brinson(
            level=level, rule="ME", method="BF", basis="period"
        )
        reporting_gap = float(pck["period_basis_max_abs_gap"])
        label_is_consistent = (
            bool(pck["reporting_view_is_exact"])
            if reporting_gap <= meaning_tol
            else not bool(pck["reporting_view_is_exact"])
        )
        rows.append({
            "block": "brinson_period_approximation_gap",
            "kind": "diagnostic",
            "value": reporting_gap,
            "tolerance": np.nan,
            "passed": True,
        })
        rows.append({
            "block": "brinson_period_reporting_semantics",
            "kind": "meaning",
            "value": 0.0 if label_is_consistent else 1.0,
            "tolerance": 0.0,
            "passed": bool(label_is_consistent),
        })

        period_r = compound_returns(eng.returns, "ME")
        wp = period_start_weights(eng.portfolio_weights, period_r)
        wb = period_start_weights(eng.benchmark_weights, period_r)
        _, s_vec, _ = brinson_multi_period(period_r, wp, wb, classifications, level, "BF")
        _, s_ref, _ = brinson_multi_period_reference(period_r, wp, wb, classifications, level, "BF")
        add(
            "brinson_vectorised_vs_reference",
            "meaning",
            (s_vec["linked_effect"] - s_ref["linked_effect"]).abs().max(),
            meaning_tol,
        )

    if factor_returns is not None:
        _, fck, _ = eng.factor_attribution(basis="native")
        add("factor_return", "identity", fck["return_additivity_error"], identity_tol)
        scale = max(abs(float(fck["portfolio_variance"])), 1e-14)
        add(
            "factor_variance",
            "identity",
            fck["variance_reconciliation_error"] / scale,
            identity_tol,
        )

    result = pd.DataFrame(rows).set_index(["block", "kind"])
    return result, bool(result["passed"].all())
