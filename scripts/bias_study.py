"""Bias study: how much do the privacy guards perturb the estimates?

Simulates Cox data (exponential hazard, two covariates, known true
coefficients) for three nodes, runs ``central`` through
``MockAlgorithmClient`` with the risk-set guard (``k``) and time binning on
and off, and reports

- the mean absolute coefficient deviation from the ``k = 1`` (guard-free)
  run on the same data,
- the mean absolute deviation from the true coefficients,
- the 95 % Wald confidence-interval coverage of the true coefficients.

Configurations: nodes per size in {200, 1k, 10k} x k in {1, 5, 10} x binning
{off, on}, each with ``--reps`` repetitions (default 50), parallelised per
configuration. NOT collected by pytest.

Run from the repository root::

    python scripts/bias_study.py --reps 50
    python scripts/bias_study.py --reps 2 --sizes 200,1000   # smoke run
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

BETA_TRUE = np.array([0.5, -0.3])
N_NODES = 3
SIZES = [200, 1000, 10000]
KS = [1, 5, 10]
BINNING_WIDTH = 5.0


def simulate(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate Cox data: exponential hazard, uniform administrative censoring."""
    X = rng.normal(size=(n_rows, 2))
    eta = X @ BETA_TRUE
    hazard = 0.02 * np.exp(eta)
    event_time = rng.exponential(1.0 / hazard)
    censor_time = rng.uniform(0.0, 60.0)
    return pd.DataFrame(
        {
            "time": np.minimum(event_time, censor_time),
            "event": (event_time <= censor_time).astype(int),
            "x1": X[:, 0],
            "x2": X[:, 1],
        }
    )


def _run_central(dfs, k: int, binning: bool):
    os.environ["COXPH_MIN_RISK_SET_CHANGE"] = str(k)
    if binning:
        os.environ["COXPH_TIME_BIN_WIDTH"] = str(BINNING_WIDTH)
    else:
        os.environ.pop("COXPH_TIME_BIN_WIDTH", None)

    from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

    datasets = [[{"database": df, "db_type": "csv"}] for df in dfs]
    client = MockAlgorithmClient(datasets=datasets, module="v6-cox-ph", organization_ids=[1, 2, 3])
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": ["x1", "x2"],
                "organization_ids": [1, 2, 3],
            },
        },
        organizations=[1],
    )
    result = client.wait_for_results(task_id=task["id"])[0]
    return result


def run_config(config):
    """One configuration (size, k, binning) over all repetitions."""
    size, k, binning, reps, seed0 = config
    devs_vs_k1 = []
    devs_vs_true = []
    coverage = []
    for rep in range(reps):
        rng = np.random.default_rng(seed0 + rep)
        dfs = [simulate(size, rng) for _ in range(N_NODES)]

        result = _run_central(dfs, k, binning)
        model = pd.DataFrame(result["model"]).T
        beta_hat = model["Coef"].to_numpy(dtype=float)
        devs_vs_true.append(float(np.max(np.abs(beta_hat - BETA_TRUE))))

        lower = np.log(model["lower_CI"].to_numpy(dtype=float))
        upper = np.log(model["upper_CI"].to_numpy(dtype=float))
        coverage.append(bool(np.all((BETA_TRUE >= lower) & (BETA_TRUE <= upper))))

        if k != 1:
            baseline = _run_central(dfs, 1, False)
            beta_k1 = pd.DataFrame(baseline["model"]).T["Coef"].to_numpy(dtype=float)
            devs_vs_k1.append(float(np.max(np.abs(beta_hat - beta_k1))))

    return {
        "size": size,
        "k": k,
        "binning": binning,
        "mean_dev_vs_k1": float(np.mean(devs_vs_k1)) if devs_vs_k1 else 0.0,
        "max_dev_vs_k1": float(np.max(devs_vs_k1)) if devs_vs_k1 else 0.0,
        "mean_dev_vs_true": float(np.mean(devs_vs_true)),
        "coverage": float(np.mean(coverage)),
        "reps": reps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=50, help="repetitions per configuration")
    parser.add_argument("--sizes", type=str, default=None, help="comma-separated node sizes (default 200,1000,10000)")
    parser.add_argument("--seed", type=int, default=20261005, help="base seed")
    parser.add_argument("--workers", type=int, default=4, help="parallel configurations")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else SIZES
    configs = [
        (size, k, binning, args.reps, args.seed + size) for size in sizes for k in KS for binning in (False, True)
    ]

    with Pool(min(args.workers, len(configs))) as pool:
        results = pool.map(run_config, configs)

    print(
        f"{'size':>7} {'k':>3} {'binning':>8} {'mean|d vs k=1|':>15} {'max|d vs k=1|':>14} "
        f"{'mean|d vs true|':>16} {'95% CI coverage':>15} {'reps':>5}"
    )
    for r in results:
        print(
            f"{r['size']:>7} {r['k']:>3} {'on' if r['binning'] else 'off':>8} "
            f"{r['mean_dev_vs_k1']:>15.4f} {r['max_dev_vs_k1']:>14.4f} "
            f"{r['mean_dev_vs_true']:>16.4f} {r['coverage']:>15.3f} {r['reps']:>5}"
        )


if __name__ == "__main__":
    main()
