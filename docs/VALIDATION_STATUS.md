# Validation status — v1.0.0

The public v1 was promoted after both synthetic/unit validation and an executed real-data notebook run.

## Automated validation

The release checks:

- package modules compile successfully;
- notebook code cells parse successfully;
- **16/16 pytest tests pass**;
- release reconciliation/meaning gate passes;
- native linked performance matches independently compounded realised return;
- native Brinson portfolio and benchmark returns match independently realised returns;
- coarser reporting-frequency Brinson exposes its approximation gap and cannot be mislabeled as exact;
- vectorised Brinson matches the slower reference implementation;
- volatility, tracking-error, Expected-Shortfall and maximum-drawdown identities reconcile within numerical tolerance;
- factor return and variance decompositions reconcile;
- `prices_to_returns` is pinned by test to avoid implicit price forward-filling.

## Real-data notebook validation

The executed public notebook was run end-to-end on adjusted public market data for the illustrative multi-asset universe.

Observed common sample in that run:

- 2010-01-05 to 2026-08-07;
- 4,173 aligned daily return observations.

The final release gate passed.

For the illustrative portfolio, native-frequency Brinson reconciled exactly to independently realised cumulative portfolio and benchmark returns.

The secondary monthly reporting view produced a non-zero approximation gap relative to the native-frequency constant-mix strategy. The engine detected that gap, labelled the monthly view as approximate, and retained the native-frequency attribution as the primary exact result.

These example outputs validate mechanics and semantics for the demonstration portfolio; they are not investment-performance claims or evidence that the supplied proxy factor specification is optimal.
