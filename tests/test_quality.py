from __future__ import annotations

from portfolio_attribution_engine import release_gate, validate_inputs


def test_clean_fixture_passes_contracts_and_gate(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    assert validate_inputs(returns, w, wb, cls, fac) == []
    table, passed = release_gate(returns, w, wb, cls, fac)
    assert passed
    assert table["passed"].all()


def test_bad_weights_fail_contracts(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    bad = wb.copy()
    bad.iloc[0] += 0.1
    issues = validate_inputs(returns, bad, wb, cls, fac)
    assert issues
    assert any("sum to one" in str(issue) for issue in issues)


def test_release_gate_checks_brinson_meaning_and_reporting_semantics(multi_asset_fixture):
    returns, w, wb, cls, fac = multi_asset_fixture
    table, passed = release_gate(returns, w, wb, cls, fac)
    assert passed

    required = {
        ("brinson_native_portfolio", "meaning"),
        ("brinson_native_benchmark", "meaning"),
        ("brinson_period_approximation_gap", "diagnostic"),
        ("brinson_period_reporting_semantics", "meaning"),
    }

    observed = set(table.index.tolist())
    assert required.issubset(observed)
    assert bool(table.loc[("brinson_period_reporting_semantics", "meaning"), "passed"])
