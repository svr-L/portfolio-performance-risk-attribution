"""One-command local release check."""
from __future__ import annotations

import compileall
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "portfolio_performance_risk_attribution.ipynb"


def fixture():
    rng = np.random.default_rng(17)
    idx = pd.bdate_range("2022-01-03", periods=378)
    cols = ["EQ", "GOVT", "CREDIT", "GOLD"]

    returns = pd.DataFrame(
        rng.standard_normal((len(idx), 4)) * 0.007 + 0.0002,
        index=idx,
        columns=cols,
    )

    w = pd.DataFrame(index=idx, columns=cols, dtype=float)
    w.iloc[:189] = [0.40, 0.30, 0.20, 0.10]
    w.iloc[189:] = [0.25, 0.35, 0.25, 0.15]

    wb = pd.Series([0.35, 0.35, 0.20, 0.10], index=cols)

    cls = pd.DataFrame(
        {"Asset Class": ["Equity", "Rates", "Credit", "Real"]},
        index=cols,
    )

    fac = pd.DataFrame(
        rng.standard_normal((len(idx), 2)) * 0.005,
        index=idx,
        columns=["MKT", "RATES"],
    )

    return returns, w, wb, cls, fac


def notebook_syntax_check(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(payload.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            compile(source, f"{path.name}:cell_{i}", "exec")
        except SyntaxError as exc:
            print(f"Notebook syntax failed in cell {i}: {exc}")
            return False
    return True


def main() -> int:
    print("1/4 Compiling package...")
    if not compileall.compile_dir(ROOT / "src", quiet=1):
        print("Package compilation failed.")
        return 1

    print("2/4 Checking notebook syntax...")
    if not notebook_syntax_check(NOTEBOOK):
        return 1

    print("3/4 Running pytest...")
    proc = subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT)
    if proc.returncode != 0:
        return proc.returncode

    print("4/4 Running reconciliation/meaning gate...")
    from portfolio_attribution_engine import release_gate

    r, w, wb, cls, fac = fixture()
    table, passed = release_gate(r, w, wb, cls, fac)
    print(table.to_string())
    print(f"\nRELEASE CHECK {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
