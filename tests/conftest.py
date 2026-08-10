from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def multi_asset_fixture():
    rng = np.random.default_rng(20260808)
    idx = pd.bdate_range("2021-01-04", periods=520)
    assets = ["EQ", "GOVT", "IG", "HY", "GOLD", "CASH"]

    # Three interpretable latent drivers plus idiosyncratic noise.
    f = rng.normal(size=(len(idx), 3))
    B = np.array([
        [0.0090, -0.0010, 0.0010],
        [0.0005,  0.0045, -0.0010],
        [0.0020,  0.0030, 0.0030],
        [0.0050, -0.0010, 0.0060],
        [0.0015,  0.0010, -0.0015],
        [0.0001,  0.0001, 0.0000],
    ])
    eps = rng.normal(scale=[0.005,0.002,0.0025,0.004,0.006,0.00005], size=(len(idx), len(assets)))
    r = f @ B.T + eps + np.array([0.00025,0.00010,0.00012,0.00018,0.00010,0.00004])
    returns = pd.DataFrame(r, index=idx, columns=assets)

    # Time-varying portfolio: one rebalance in the middle.
    w = pd.DataFrame(index=idx, columns=assets, dtype=float)
    w.iloc[:260] = [0.45, 0.20, 0.12, 0.08, 0.10, 0.05]
    w.iloc[260:] = [0.35, 0.25, 0.13, 0.07, 0.15, 0.05]
    wb = pd.Series([0.40, 0.25, 0.15, 0.05, 0.10, 0.05], index=assets)
    cls = pd.DataFrame({
        "Asset Class": ["Equity","Rates","Credit","Credit","Real Assets","Cash"],
        "Risk Sleeve": ["Growth","Duration","Duration","Credit","Diversifier","Liquidity"],
    }, index=assets)
    fac = pd.DataFrame({
        "Equity": f[:,0] * 0.007,
        "Rates": f[:,1] * 0.004,
        "Credit": f[:,2] * 0.005,
    }, index=idx)
    return returns, w, wb, cls, fac
