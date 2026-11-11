"""
Reference (pre-vectorisation) risk-set implementation, for equivalence tests.

This is the mask-based implementation ``perform_iteration`` used before the
``searchsorted``/``bincount`` rewrite, kept verbatim as the
reference that ``guarded_risk_set_aggregates`` must reproduce to rtol=1e-12.
"""

import numpy as np
import pandas as pd


def guarded_risk_set_masks(times: pd.Series, grid: list[float], k: int) -> list[np.ndarray]:
    """Compute risk-set masks with the minimum-change ("jump") guard.

    Walking the grid from smallest to largest time, if moving from ``t_i`` to
    ``t_{i+1}`` would remove fewer than ``k`` (but more than zero)
    individuals from the risk set, the previous (larger) risk set is held.
    Consequently consecutive shared aggregates differ by 0 or by at least
    ``k`` individuals. When ``k <= 1`` the guard is a no-op and the masks are
    the plain ``times >= t`` masks.

    Returns
    -------
    list[np.ndarray]
        One boolean mask per grid point (aligned with ``grid``).
    """
    times_arr = pd.to_numeric(times, errors="coerce").to_numpy()

    if k is None or k <= 1:
        return [times_arr >= t for t in grid]

    masks: list[np.ndarray] = []
    current = times_arr >= grid[0]
    masks.append(current)
    for t in grid[1:]:
        cand = times_arr >= t
        removed = int(current.sum() - cand.sum())
        if 0 < removed < k:
            # Hold the previous risk set.
            masks.append(current)
        else:
            current = cand
            masks.append(current)
    return masks


def mask_based_aggregates(times, grid, k, X, beta):
    """The aggregates the mask implementation produced, per grid point."""
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    p = X.shape[1]
    masks = guarded_risk_set_masks(times, grid, k)

    agg1: list = []
    agg2: list = []
    agg3: list = []
    for mask in masks:
        n_in_set = int(mask.sum())
        if n_in_set == 0:
            agg1.append(0)
            agg2.append(np.zeros(p))
            agg3.append(np.zeros((p, p)))
        else:
            X_m = X[mask]
            ebz = np.exp(X_m @ beta)
            agg1.append(float(ebz.sum()))
            agg2.append((X_m * ebz[:, None]).sum(axis=0))
            agg3.append((X_m * ebz[:, None]).T @ X_m)
    return np.array(agg1), np.array(agg2), np.array(agg3)
