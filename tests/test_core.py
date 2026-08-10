from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_attribution_engine import (
    PortfolioAttributionEngine,
    brinson_multi_period,
    brinson_multi_period_reference,
    brinson_single_period,
    carino_link,
    compound_returns,
    expected_shortfall_tail_weights,
    prices_to_returns,
    scenario_expected_shortfall,
)


def test_prices_to_returns_does_not_silently_forward_fill():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100.0, np.nan, 102.0, 103.0]}, index=idx)
    r = prices_to_returns(prices, missing="drop")
    assert np.isnan(r.loc[idx[1], "A"])
    assert np.isnan(r.loc[idx[2], "A"])
    assert np.isclose(r.loc[idx[3], "A"], 103/102 - 1)


def test_native_linking_matches_independent_realised_return(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    eng = PortfolioAttributionEngine(returns, w, wb, cls, fac)
    table, check = eng.linked_asset_performance(basis="native")
    realised = float((1 + (w * returns).sum(axis=1)).prod() - 1)
    assert abs(check["linking_error"]) < 1e-12
    assert abs(table["linked_return_contribution"].sum() - realised) < 1e-12


def test_native_brinson_matches_independent_realised_portfolio_and_benchmark(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    eng = PortfolioAttributionEngine(returns, w, wb, cls, fac)

    _, _, check = eng.brinson(
        level="Asset Class",
        method="BF",
        basis="native",
    )

    realised_portfolio = float((1.0 + (w * returns).sum(axis=1)).prod() - 1.0)
    realised_benchmark = float((1.0 + (returns * wb).sum(axis=1)).prod() - 1.0)

    assert abs(check["cumulative_portfolio_return"] - realised_portfolio) < 1e-12
    assert abs(check["cumulative_benchmark_return"] - realised_benchmark) < 1e-12
    assert abs(check["linking_error"]) < 1e-12


def test_period_brinson_exposes_constant_mix_approximation_gap(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture

    # A static target vector is still a native-frequency constant-mix strategy:
    # the same target weights are applied to every native return observation.
    static_wp = w.iloc[0].copy()

    eng = PortfolioAttributionEngine(returns, static_wp, wb, cls, fac)

    _, _, check = eng.brinson(
        level="Asset Class",
        method="BF",
        basis="period",
        rule="ME",
    )

    assert check["period_basis_max_abs_gap"] > 1e-8
    assert not bool(check["reporting_view_is_exact"])
    assert "approximate" in check["reporting_view_status"]
    assert abs(
        check["exact_native_cumulative_portfolio_return"]
        - float((1.0 + eng.portfolio_returns).prod() - 1.0)
    ) < 1e-12


def test_period_view_discloses_intra_period_rebalancing(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    # Put a rebalance inside a reporting month to force the disclosure.
    w2 = w.copy()
    split = 245
    w2.iloc[split:] = [0.30, 0.30, 0.15, 0.05, 0.15, 0.05]
    eng = PortfolioAttributionEngine(returns, w2, wb, cls, fac)
    _, check = eng.linked_asset_performance(rule="ME", basis="period")
    assert bool(check["intra_period_rebalancing"])
    assert "period_basis_approximation_error" in check.index


def test_brinson_single_period_reconciles_bhb_and_bf():
    assets = pd.Index(["A", "B", "C", "D"])
    r = pd.Series([0.10, 0.04, -0.02, 0.03], index=assets)
    wp = pd.Series([0.35, 0.25, 0.25, 0.15], index=assets)
    wb = pd.Series([0.25, 0.25, 0.30, 0.20], index=assets)
    cls = pd.DataFrame({"Sector": ["X", "X", "Y", "Y"]}, index=assets)
    for method in ["BHB", "BF"]:
        _, ck = brinson_single_period(wp, wb, r, cls, level="Sector", method=method)
        assert abs(ck["additivity_error"]) < 1e-14


def test_vectorised_brinson_matches_reference(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    period_r = compound_returns(returns, "ME")
    # Static weights here make the comparison intentionally simple.
    wp = w.iloc[0]
    for method in ["BHB", "BF"]:
        d1, s1, c1 = brinson_multi_period(period_r, wp, wb, cls, "Asset Class", method)
        d2, s2, c2 = brinson_multi_period_reference(period_r, wp, wb, cls, "Asset Class", method)
        assert np.allclose(d1.select_dtypes("number"), d2.select_dtypes("number"), atol=1e-13)
        assert np.allclose(s1, s2, atol=1e-13)
        assert abs(c1["linking_error"]) < 1e-12
        assert abs(c2["linking_error"]) < 1e-12


def test_carino_links_to_cumulative_active_return():
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    rp = pd.Series([0.02, -0.01, 0.03], index=idx)
    rb = pd.Series([0.01, 0.00, 0.02], index=idx)
    effects = pd.DataFrame({
        "allocation": [0.004, -0.003, 0.004],
        "selection": [0.004, -0.004, 0.004],
        "interaction": [0.002, -0.003, 0.002],
    }, index=idx)
    # Each row must equal period active return.
    assert np.allclose(effects.sum(axis=1), rp-rb)
    linked, ck = carino_link(effects, rp, rb)
    assert abs(linked.sum() - ck["cumulative_active_return"]) < 1e-12


def test_factor_return_and_variance_reconcile(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    eng = PortfolioAttributionEngine(returns, w, wb, cls, fac)
    _, ck, residual = eng.factor_attribution(basis="native")
    assert abs(ck["return_additivity_error"]) < 1e-12
    assert abs(ck["variance_reconciliation_error"]) < 1e-12
    assert len(residual) == len(returns)
    assert 0 <= ck["weighted_R2"] <= 1


def test_volatility_tracking_error_and_es_reconcile(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    eng = PortfolioAttributionEngine(returns, w, wb, cls, fac)
    _, vck = eng.volatility(lookback=252)
    _, tck = eng.tracking_error(lookback=252)
    _, eck, _ = eng.expected_shortfall(alpha=0.975, lookback=252)
    assert abs(vck["volatility_additivity_error"]) < 1e-12
    assert abs(tck["tracking_error_additivity_error"]) < 1e-12
    assert abs(eck["additivity_error"]) < 1e-12


def test_fractional_expected_shortfall_boundary():
    losses = np.array([1., 2., 3., 4.])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    var, es, tw = expected_shortfall_tail_weights(losses, p, alpha=0.75)
    assert np.isclose(tw.sum(), 1.0)
    assert np.isclose((tw > 0).sum(), 1)  # worst scenario alone has more than 25% mass
    assert np.isclose(es, 4.0)


def test_scenario_adapter_is_same_es_engine():
    scenarios = pd.DataFrame({
        "EQ": [-0.10, -0.04, 0.02, 0.05],
        "BOND": [0.02, -0.01, 0.01, 0.00],
    })
    w = pd.Series({"EQ": 0.6, "BOND": 0.4})
    p = np.array([0.15, 0.20, 0.35, 0.30])
    tbl, ck, tw = scenario_expected_shortfall(scenarios, w, p, alpha=0.80, portfolio_value=1_000_000)
    assert abs(tbl["component_ES"].sum() - ck["Expected_Shortfall"]) < 1e-8
    assert np.isclose(tw.sum(), 1.0)


def test_max_drawdown_reconciles(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    eng = PortfolioAttributionEngine(returns, w, wb, cls, fac)
    _, ck = eng.max_drawdown()
    assert abs(ck["additivity_error"]) < 1e-12
