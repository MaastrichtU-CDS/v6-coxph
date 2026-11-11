"""Benchmark: mask-based vs vectorised risk-set aggregates.

Times and measures peak memory of both implementations on
100,000 rows x 10,000 event times x 10 covariates (k = 5), the
configuration recorded in docs/coxph/Validation.rst.

Run from the repository root::

    python scripts/benchmark_risk_sets.py
"""

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "v6-cox-ph"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "unit"))

from privacy_guards import guarded_risk_set_aggregates  # noqa: E402
from reference_risk_sets import mask_based_aggregates  # noqa: E402

N_ROWS = 100_000
N_GRID = 10_000
N_COVS = 10
K = 5


def bench(func, *args):
    start = time.perf_counter()
    tracemalloc.start()
    result = func(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return time.perf_counter() - start, peak / (1024 * 1024), result


def main():
    rng = np.random.default_rng(42)
    grid = np.sort(rng.uniform(0.0, 100.0, size=N_GRID)).tolist()
    times = pd.Series(rng.uniform(0.0, 100.0, size=N_ROWS))
    X = rng.normal(size=(N_ROWS, N_COVS))
    beta = rng.normal(size=N_COVS) * 0.1

    print(f"{N_ROWS} rows x {N_GRID} event times x {N_COVS} covariates, k={K}")
    t_mask, m_mask, _ = bench(mask_based_aggregates, times, grid, K, X, beta)
    print(f"mask-based : {t_mask:8.2f} s, peak {m_mask:8.1f} MB")
    t_vec, m_vec, _ = bench(guarded_risk_set_aggregates, times, grid, K, X, beta)
    print(f"vectorised : {t_vec:8.2f} s, peak {m_vec:8.1f} MB")
    print(f"speed-up   : {t_mask / t_vec:8.1f} x, memory ratio {m_mask / max(m_vec, 1e-9):.1f} x")


if __name__ == "__main__":
    main()
