# Methods and conventions

## 1. Absolute performance attribution

For asset weights `w_{t,i}` and simple asset returns `r_{t,i}`, the native-frequency portfolio return is

`r_{p,t} = Σ_i w_{t,i} r_{t,i}`.

The period contribution of asset `i` is

`c_{t,i} = w_{t,i} r_{t,i}`.

Single-period contributions are additive, but simple returns are not additive through time. The engine therefore geometrically links native-period contributions so that the final asset contributions sum exactly to

`Π_t (1 + r_{p,t}) - 1`.

This is the preferred absolute-performance attribution because it remains exact under static or time-varying native-frequency weights.

## 2. Weight timing and reporting-frequency attribution

At the native return frequency, each weight row is interpreted as the target capital allocation applied to that observation.

A static target-weight vector is therefore a native-frequency constant-mix portfolio. With daily returns, the engine applies the target weights at each daily return observation.

This distinction matters when returns are first compounded to a coarser reporting frequency.

The engine distinguishes:

- `basis="native"` — exact attribution for the realised strategy implied by the native-frequency weights;
- `basis="period"` — a reporting view in which asset returns are compounded to a requested frequency and beginning-of-period weights are used.

The period view can represent a different effective rebalancing convention even if the target weights do not numerically change inside the reporting period. For example, daily constant-mix weights and one monthly weight vector generally do not generate the same compounded monthly portfolio return.

The engine therefore compares the reporting view with native-frequency realised performance and reports:

- `period_basis_portfolio_gap`;
- `period_basis_benchmark_gap`;
- `period_basis_active_gap`;
- `period_basis_max_abs_gap`;
- `reporting_view_is_exact`;
- `reporting_view_status`.

A non-zero gap is a diagnostic, not an internal Brinson arithmetic error.

## 3. Brinson attribution

For each group `g`, the engine calculates portfolio/benchmark group weights and group returns.

It supports:

- **BHB allocation**: `(w_p,g - w_b,g) r_b,g`;
- **BF allocation**: `(w_p,g - w_b,g) (r_b,g - r_b)`;
- **selection**: `w_b,g (r_p,g - r_b,g)`;
- **interaction**: `(w_p,g - w_b,g) (r_p,g - r_b,g)`.

The sum of allocation, selection and interaction reconciles to single-period active return.

Across periods, Carino linking reconciles linked effects to cumulative portfolio return minus cumulative benchmark return.

### Primary versus secondary Brinson views

The public v1 treats native-frequency Brinson as the primary result because it corresponds to the same portfolio path used to compute realised portfolio and benchmark returns.

A coarser reporting-period Brinson view remains useful for conventional investment reporting, but its economic equivalence to the native-frequency strategy is checked rather than assumed.

## 4. Factor regression attribution

The public v1 estimates a probability-weighted portfolio regression

`r_p = α + β' f + ε`.

Expected portfolio return is decomposed into:

- intercept `α`;
- `β_k E[f_k]` for each supplied factor;
- mean residual.

Portfolio variance is decomposed into Euler factor components

`β_k (Σ_f β)_k`

plus residual variance.

This is a **portfolio-level regression decomposition**. It is not a security-level factor covariance model and should not be described as one.

## 5. Volatility attribution

For portfolio weights `w` and covariance matrix `Σ`,

`σ_p² = w' Σ w`.

Component variance is

`w_i (Σw)_i`.

Dividing component variance by portfolio volatility gives component volatility. Contributions can be negative when an exposure hedges or diversifies the rest of the portfolio.

## 6. Tracking-error attribution

With active weights

`a = w_p - w_b`,

tracking-error variance is

`TE² = a' Σ a`.

The Euler component is

`a_i (Σa)_i / TE`,

which sums to total tracking error.

## 7. Expected Shortfall attribution

For scenario asset returns `R_{j,i}` and portfolio weights `w_i`, asset loss contribution is

`L_{j,i} = -w_i R_{j,i} V`.

Portfolio loss is

`L_j = Σ_i L_{j,i}`.

The engine constructs exactly the worst `(1-α)` probability mass, including a fractional boundary scenario if required.

Component ES is the probability-weighted mean of each `L_{j,i}` using the **same tail weights** as portfolio ES, so components add exactly to total ES.

## 8. Drawdown and stress attribution

For a user-defined stress window, native contributions are geometrically linked inside the selected window.

For maximum drawdown, the engine identifies the peak-to-trough episode of the realised portfolio wealth path and attributes the cumulative loss over that exact episode to assets.

This is a realised-path decomposition, not a forecast of future drawdown.

## 9. Data hygiene

- simple returns are required;
- returns at or below `-100%` are invalid for geometric linking;
- missing prices are never silently forward-filled by default;
- `pct_change(fill_method=None)` is used explicitly;
- complete-case alignment can shorten the common sample, so `returns_coverage` should be reviewed;
- portfolio and benchmark weights must sum to one;
- classifications must cover every asset used in the requested hierarchy.

## 10. Validation philosophy

The engine distinguishes mathematical reconciliation from economic meaning.

A decomposition that sums to its own internally constructed total is not sufficient if that total represents a different portfolio path.

The release gate therefore verifies, among other checks:

- native linked performance versus independently realised portfolio return;
- native Brinson portfolio and benchmark returns versus independently realised returns;
- consistency of the exact/approximate label for coarser reporting views;
- vectorised Brinson versus a slower reference implementation;
- factor return/variance reconciliation;
- volatility, tracking-error, ES and drawdown additivity.
