# Portfolio Performance & Risk Attribution Engine

A modular Python engine for **absolute performance attribution**, **benchmark-relative Brinson attribution**, **portfolio factor diagnostics**, and **portfolio-risk attribution** across multi-asset portfolios.

The project is designed as an attribution/diagnostics layer: it does **not** generate alpha signals, optimise portfolios, or simulate markets. Instead, it consumes portfolio returns, weights, benchmarks, classifications, factors, or scenario matrices produced elsewhere and explains **where performance and risk came from**.

## Why this project exists

Portfolio construction, scenario generation and overlays answer questions such as:

- What portfolio should I hold?
- What scenarios should I consider?
- How should exposures change?

This engine answers the complementary questions:

- Which assets or sleeves generated realised performance?
- Was active return driven by allocation, selection or interaction?
- Which factors explain portfolio return variation?
- Which holdings consume volatility, tracking-error and Expected-Shortfall budgets?
- Which positions drove the maximum drawdown or a user-defined stress window?
- How can an external scenario engine decompose forward-looking tail risk?

## Public v1 scope

### Performance attribution

- single-period asset contribution;
- exact native-frequency multi-period asset linking;
- reporting-period attribution with explicit intra-period-rebalancing diagnostics;
- Brinson–Hood–Beebower (BHB);
- Brinson–Fachler (BF);
- Carino multi-period linking;
- hierarchical attribution through user-supplied classifications.

### Factor diagnostics

- probability-weighted portfolio regression on user-supplied factors;
- expected-return contribution by factor plus intercept/residual;
- factor-explained versus residual portfolio variance;
- explicit reconciliation checks.

This is a **portfolio-level regression attribution**, not a security-level factor-risk model.

### Risk attribution

- Euler volatility contribution;
- Euler tracking-error contribution versus a benchmark;
- historical or scenario Expected Shortfall contribution;
- probability-weighted ES with fractional boundary-scenario handling;
- maximum-drawdown contribution;
- user-defined stress-period contribution.

### Integration

- static or time-varying portfolio weights;
- configurable portfolio hierarchies (asset class, region, risk sleeve, currency, etc.);
- scenario Expected-Shortfall adapter for external scenario generators;
- release gate checking both mathematical identities and meaning-level invariants.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[demo,test]"
```

Run the test suite:

```bash
pytest
```

Run the full local release check:

```bash
python scripts/release_check.py
```

Run the real multi-asset demo:

```bash
python examples/real_multi_asset_demo.py
```

## Real-data example

`examples/real_multi_asset_demo.py` downloads adjusted prices for a transparent public ETF universe:

- `VT` — global equity;
- `IEF` — US Treasuries;
- `LQD` — investment-grade credit;
- `HYG` — high-yield credit;
- `VNQ` — listed real estate;
- `GLD` — gold;
- `DBC` — broad commodities;
- `BIL` — Treasury bills / cash proxy.

Portfolio and benchmark weights plus hierarchy labels live in `examples/portfolio_definition.csv`. They are demonstration inputs, not investment recommendations.

The demo uses three transparent daily proxy factors (`VT-BIL`, `IEF-BIL`, `HYG-IEF`) only to exercise the factor interface. Replace them with the factor library appropriate to the research problem.

## Minimal API

```python
import pandas as pd
from portfolio_attribution_engine import PortfolioAttributionEngine

engine = PortfolioAttributionEngine(
    returns=asset_returns,
    portfolio_weights=portfolio_weights,
    benchmark_weights=benchmark_weights,
    classifications=classifications,
    factor_returns=factor_returns,
)

asset_table, asset_check = engine.linked_asset_performance(basis="native")
_, brinson_summary, brinson_check = engine.brinson(
    level="Asset Class", method="BF", basis="period", rule="ME"
)
factor_table, factor_check, residual = engine.factor_attribution(basis="native")
vol_table, vol_check = engine.volatility(lookback=756)
te_table, te_check = engine.tracking_error(lookback=756)
es_table, es_check, tail_weights = engine.expected_shortfall(alpha=0.975, lookback=756)
dd_table, dd_check = engine.max_drawdown()
```

For forward-looking scenarios generated elsewhere:

```python
from portfolio_attribution_engine import scenario_expected_shortfall

scenario_table, scenario_check, tail_weights = scenario_expected_shortfall(
    scenario_returns,
    weights,
    probabilities=scenario_probabilities,
    alpha=0.975,
    portfolio_value=10_000_000,
)
```

## Important conventions

### Returns

Inputs are **simple returns**, not prices. If starting from price/total-return-index data, use `prices_to_returns`. Missing prices are **not** implicitly forward-filled: `pct_change(fill_method=None)` is used explicitly.

### Weights

At the native frequency, each weight row is interpreted as the capital allocation **in force for that return observation**. Time-varying weights are forward-filled from supplied rebalance dates.

For coarser reporting periods, the engine uses beginning-of-period weights. If weights change inside a reporting period, the period-frequency view is necessarily an approximation; the exact native-frequency result is retained as the benchmark.

### Brinson

BHB and BF are both available. The engine reports allocation, selection and interaction separately and Carino-links multi-period effects to cumulative active return.

### Expected Shortfall

Asset losses are formed scenario-by-scenario and component ES uses the **same portfolio-tail weights** as total ES. Therefore component contributions add exactly to portfolio ES, including when the ES probability boundary cuts through a scenario with non-zero probability mass.

## Validation philosophy

A decomposition can add up perfectly and still represent the wrong economic object. The project therefore tests two classes of invariants:

1. **identity checks** — components sum to the relevant portfolio total;
2. **meaning checks** — for example, linked performance must equal realised cumulative portfolio return computed independently, and vectorised Brinson must match the slower reference implementation.

`release_gate(...)` is intended as the final check before producing public output.

## Repository structure

```text
src/portfolio_attribution_engine/
    core.py          # stable attribution algorithms
    adapters.py      # scenario-engine interface
    quality.py       # input contracts + release gate
examples/
    portfolio_definition.csv
    real_multi_asset_demo.py
notebooks/
    real_multi_asset_attribution_demo.ipynb
tests/
    test_core.py
    test_quality.py
docs/
    METHODS.md
    INTERVIEW_NOTES.md
    CV_BULLETS.md
    RELEASE_CHECKLIST.md
```

## What public v1 deliberately does not claim

The following are useful future extensions but are **not part of this release**:

- security-level factor covariance models (`B Ω B' + D`);
- regularised factor selection;
- factor-level Expected Shortfall attribution;
- currency attribution;
- fixed-income carry/rolldown/curve/spread attribution;
- Greeks-based derivatives P&L explain;
- rolling/counterfactual decision layers;
- statistical inference for attribution estimates.

The public release is intentionally smaller than the development roadmap so every advertised feature is present, tested and defensible.

## Disclaimer

Research/educational software. Not investment advice and not a production risk-management system.
