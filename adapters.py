"""End-to-end real-data demo using liquid public ETF proxies.

Run from the repository root after:
    pip install -e '.[demo]'
    python examples/real_multi_asset_demo.py

The example universe is transparent and replaceable. It is not an investment
recommendation and the proxy factors are not a canonical factor model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from portfolio_attribution_engine import (
    PortfolioAttributionEngine,
    prices_to_returns,
    release_gate,
    returns_coverage,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = pd.read_csv(ROOT / "examples" / "portfolio_definition.csv").set_index("ticker")
TICKERS = CONFIG.index.tolist()
START = "2010-01-01"


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No market data returned by yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Expected adjusted Close prices in yfinance output.")
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return px.reindex(columns=tickers).sort_index()


prices = download_prices(TICKERS, START)
returns_raw = prices_to_returns(prices, missing="drop")

print("\nDATA COVERAGE BEFORE COMPLETE-CASE ALIGNMENT")
print(returns_coverage(returns_raw).to_string())

returns = returns_raw.dropna(how="any")

wp = CONFIG["portfolio_weight"].astype(float)
wb = CONFIG["benchmark_weight"].astype(float)
classifications = CONFIG[["Asset Class", "Risk Sleeve", "Region", "Currency"]].copy()

# Transparent investable proxy factors at native daily frequency.
# Replace these with the factor library appropriate to the research problem.
factors = pd.DataFrame(index=returns.index)
factors["Equity_ex_cash"] = returns["VT"] - returns["BIL"]
factors["Duration_ex_cash"] = returns["IEF"] - returns["BIL"]
factors["Credit_ex_rates"] = returns["HYG"] - returns["IEF"]

engine = PortfolioAttributionEngine(
    returns=returns,
    portfolio_weights=wp,
    benchmark_weights=wb,
    classifications=classifications,
    factor_returns=factors,
)

print("\nABSOLUTE MULTI-PERIOD PERFORMANCE ATTRIBUTION")
asset_perf, asset_check = engine.linked_asset_performance(basis="native")
print(asset_perf.sort_values("linked_return_contribution", ascending=False).to_string())
print(asset_check.to_string())

for level in ["Asset Class", "Risk Sleeve"]:
    print(f"\nEXACT NATIVE BRINSON-FACHLER — {level.upper()}")
    _, summary, check = engine.brinson(
        level=level,
        method="BF",
        basis="native",
    )
    print(summary.to_string())
    print(check.to_string())

    print(f"\nSECONDARY MONTHLY REPORTING VIEW — {level.upper()}")
    _, monthly_summary, monthly_check = engine.brinson(
        level=level,
        method="BF",
        basis="period",
        rule="ME",
    )
    print(monthly_summary.to_string())
    print(monthly_check.to_string())

print("\nFACTOR REGRESSION ATTRIBUTION")
factor_table, factor_check, _ = engine.factor_attribution(basis="native")
print(factor_table.to_string())
print(factor_check.to_string())

print("\nVOLATILITY ATTRIBUTION — LAST 756 BUSINESS DAYS")
vol_table, vol_check = engine.volatility(lookback=756)
print(vol_table.sort_values("component_volatility", ascending=False).to_string())
print(vol_check.to_string())

print("\nTRACKING-ERROR ATTRIBUTION — LAST 756 BUSINESS DAYS")
te_table, te_check = engine.tracking_error(lookback=756)
print(te_table.sort_values("component_tracking_error", ascending=False).to_string())
print(te_check.to_string())

print("\nEXPECTED SHORTFALL ATTRIBUTION — 97.5%, LAST 756 BUSINESS DAYS")
es_table, es_check, _ = engine.expected_shortfall(alpha=0.975, lookback=756)
print(es_table.sort_values("component_ES", ascending=False).to_string())
print(es_check.to_string())

print("\nMAXIMUM-DRAWDOWN ATTRIBUTION")
dd_table, dd_check = engine.max_drawdown()
print(dd_table.sort_values("drawdown_contribution", ascending=False).to_string())
print(dd_check.to_string())

print("\nRELEASE GATE")
gate, passed = release_gate(returns, wp, wb, classifications, factors)
print(gate.to_string())
print(f"\nRELEASE GATE: {'PASSED' if passed else 'FAILED'}")

if not passed:
    raise SystemExit(1)
