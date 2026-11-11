"""
Cox Proportional Hazards mathematical logic.

This module contains the core mathematical computations for the Cox-PH model,
separated from the orchestration logic in central.py and partial.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from vantage6.algorithm.tools.exceptions import AlgorithmError
from vantage6.algorithm.tools.util import info, warn

from typing import Any, Dict, List, Optional, Tuple


def compute_derivatives(
    summed_agg1: np.ndarray,
    summed_agg2: np.ndarray,
    summed_agg3: np.ndarray,
    aggregated_time_events: pd.DataFrame,
    z_sum: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the primary and secondary derivatives for the Cox-PH model.

    These derivatives are used in the Newton-Raphson optimisation to find
    the maximum partial likelihood estimates of the regression coefficients.

    Parameters
    ----------
    summed_agg1 : np.ndarray
        Aggregated sum of exp(beta * X) across all nodes for each event time.
    summed_agg2 : np.ndarray
        Aggregated sum of X * exp(beta * X) across all nodes for each event time.
    summed_agg3 : np.ndarray
        Aggregated sum of outer products X * X^T * exp(beta * X) across all
        nodes for each event time.
    aggregated_time_events : pd.DataFrame
        DataFrame containing unique event times and their frequencies.
    z_sum : pd.Series
        Sum of explanatory variables for all event cases.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - primary_derivative: First derivative of the partial log-likelihood.
        - secondary_derivative: Second derivative (Hessian) of the partial
          log-likelihood.
    """
    info("Computing derivatives for Cox-PH model")

    n_covs = len(z_sum)
    n_times = len(aggregated_time_events)
    tot_p1 = np.zeros(n_covs)
    tot_p2 = np.zeros((n_covs, n_covs))

    freqs = aggregated_time_events["freq"].to_numpy()

    if len(summed_agg1) != n_times:
        raise AlgorithmError(
            f"Length mismatch: aggregated_time_events has {n_times} rows but "
            f"summed_agg1 has {len(summed_agg1)} entries"
        )

    for pos in range(n_times):
        freq = freqs[pos]

        s1_value = summed_agg1[pos]
        s2_value = np.asarray(summed_agg2[pos])
        s3_value = np.asarray(summed_agg3[pos])

        if not (s1_value > 0) or np.isnan(s1_value):
            raise AlgorithmError(
                f"Risk-set sum S0 is not positive at grid position {pos}; "
                f"the aggregated risk sets are inconsistent."
            )

        # Primary derivative component
        s1 = freq * (s2_value / s1_value)

        # Secondary derivative component
        first_part = s3_value / s1_value
        numerator = np.outer(s2_value, s2_value)
        denominator = s1_value * s1_value
        second_part = numerator / denominator

        s2 = freq * (first_part - second_part)

        tot_p1 += s1
        tot_p2 += s2

    primary_derivative = z_sum.values - tot_p1
    secondary_derivative = -tot_p2

    return primary_derivative, secondary_derivative


def compute_model_results(
    beta: np.ndarray,
    secondary_derivative: np.ndarray,
    z_sum: pd.Series,
    aggregated_time_events: pd.DataFrame,
    summed_agg1: np.ndarray,
    expl_vars: List[str],
    converged: bool = True,
    n_iterations: int | None = None,
) -> Dict[str, Any]:
    """
    Compute final model results after convergence.

    Parameters
    ----------
    beta : np.ndarray
        Final estimated regression coefficients.
    secondary_derivative : np.ndarray
        Final Hessian matrix (negative second derivative), evaluated at the
        same ``beta`` that is reported.
    z_sum : pd.Series
        Sum of explanatory variables for all event cases.
    aggregated_time_events : pd.DataFrame
        DataFrame containing unique event times and their frequencies.
    summed_agg1 : np.ndarray
        Final aggregated sum of exp(beta * X) for each event time, evaluated
        at the same ``beta`` that is reported.
    expl_vars : List[str]
        Names of the explanatory variables.
    converged : bool
        Whether the Newton-Raphson optimiser converged. When ``False`` a
        warning is appended so users know SE/p-values may be unreliable.
    n_iterations : int | None
        Number of Newton-Raphson iterations performed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model coefficients, statistics, and warnings.
    """
    info("Computing final model results")

    n_covs = len(beta)

    # Standard errors from the covariance matrix (inverse of the observed
    # Fisher information, i.e. the inverse of the negative Hessian).
    try:
        covariance = np.linalg.inv(-secondary_derivative)
        serrors = np.array([np.sqrt(covariance[k, k]) for k in range(covariance.shape[0])])
    except np.linalg.LinAlgError as e:
        warn(f"Could not invert Hessian matrix: {e}")
        covariance = np.zeros((n_covs, n_covs))
        serrors = np.array([np.nan] * n_covs)

    # Z-values and p-values (Wald test: Z = beta / SE)
    with np.errstate(divide="ignore", invalid="ignore"):
        zvalues = np.zeros(n_covs)
        pvalues = np.ones(n_covs)
        for i in range(n_covs):
            if serrors[i] > 0 and not np.isnan(serrors[i]):
                zvalues[i] = beta[i] / serrors[i]
                pvalues[i] = 2 * norm.cdf(-abs(zvalues[i]))
            else:
                zvalues[i] = np.nan
                pvalues[i] = np.nan

    # Overall model significance (Wald test)
    degrees_of_freedom = len(beta)
    try:
        wald_statistic = np.dot(beta, np.dot(-secondary_derivative, beta))
        overall_p_value = float(chi2.sf(wald_statistic, degrees_of_freedom))
    except (ValueError, FloatingPointError) as e:
        warn(f"Could not compute Wald statistic: {e}")
        overall_p_value = None

    # AIC for model comparison
    try:
        linear_part = np.dot(z_sum.values, beta)

        risk_set_part = 0
        freqs = aggregated_time_events["freq"].to_numpy()
        if hasattr(summed_agg1, "__len__") and len(summed_agg1) > 0:
            for i in range(len(freqs)):
                if i < len(summed_agg1) and summed_agg1[i] > 0:
                    risk_set_part += freqs[i] * np.log(summed_agg1[i])

        log_likelihood = linear_part - risk_set_part
        n_params = len(beta)

        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")

        aic = float(-2 * log_likelihood + 2 * n_params)
    except (ValueError, IndexError, FloatingPointError) as e:
        warn(f"Could not compute AIC due to numerical/data issue: {e}")
        aic = None

    # Results data — full precision; rounding is a client concern.
    results_data = {
        "Coef": beta,
        "Exp(coef)": np.exp(beta),
        "SE": serrors,
        "lower_CI": np.exp(beta - 1.96 * serrors),
        "upper_CI": np.exp(beta + 1.96 * serrors),
        "Z": zvalues,
        "p-value": pvalues,
    }

    # Perfect prediction warnings
    warnings = []
    threshold = 10
    for i, (coef, se) in enumerate(zip(beta, serrors)):
        if (
            abs(coef) > threshold
            or np.isinf(coef)
            or np.isnan(coef)
            or abs(se) > threshold
            or np.isinf(se)
            or np.isnan(se)
        ):
            var_name = expl_vars[i] if i < len(expl_vars) else f"var_{i}"
            msg = (
                f"Warning: Covariate '{var_name}' may perfectly predict the event "
                f"(coef={coef}, SE={se}). Results may be unreliable."
            )
            warn(msg)
            warnings.append(msg)

    if not converged:
        n_iter_str = str(n_iterations) if n_iterations is not None else "unknown"
        msg = (
            f"Newton-Raphson did not converge in {n_iter_str} iterations; "
            f"SE/p-values may be unreliable; statistics are reported at the "
            f"last evaluated beta"
        )
        warnings.append(msg)

    return {
        "results_data": results_data,
        "covariance": covariance,
        "standard_errors": serrors,
        "zvalues": zvalues,
        "pvalues": pvalues,
        "overall_p_value": overall_p_value,
        "aic": aic,
        "n_params": int(len(beta)),
        "warnings": warnings,
    }


def partial_log_likelihood(
    beta: np.ndarray,
    z_sum: "pd.Series | np.ndarray",
    aggregated_time_events: pd.DataFrame,
    summed_agg1: np.ndarray,
) -> float:
    """
    The Breslow partial log-likelihood at ``beta``, from pooled aggregates.

    ``ell(beta) = beta . z_sum - sum_j d_j * log S0(t_j)`` where ``d_j`` is the
    pooled event count at grid time ``t_j`` and ``S0(t_j)`` the aggregated
    risk-set sum at the same ``beta``. Everything it needs is already
    collected, so no extra round-trip is required.

    Raises
    ------
    AlgorithmError
        If any risk-set sum is not positive or not finite.
    """
    beta = np.asarray(beta, dtype=float)
    s0 = np.asarray(summed_agg1, dtype=float)
    if len(s0) != len(aggregated_time_events):
        raise AlgorithmError(
            f"Length mismatch: aggregated_time_events has {len(aggregated_time_events)} "
            f"rows but summed_agg1 has {len(s0)} entries"
        )
    if np.any(s0 <= 0) or not np.all(np.isfinite(s0)):
        raise AlgorithmError("A risk-set sum S0 is not positive; cannot evaluate the log-likelihood.")
    freqs = aggregated_time_events["freq"].to_numpy(dtype=float)
    return float(np.dot(beta, np.asarray(z_sum, dtype=float)) - np.sum(freqs * np.log(s0)))


def held_event_fraction(agg1, freq, min_risk_set_change: int) -> float:
    """
    The fraction of a node's events scored against a held (unchanged) risk set.

    ``agg1`` is the node's released risk-set sizes on the event-time grid at
    ``beta = 0``, where they are exact integers under the released
    precision, and ``freq`` the node's per-time event counts on the same
    grid. A grid point beyond the first is *held* when the released risk
    set is unchanged since the first release (``agg1[i] == agg1[0]``): the
    true risk set strictly decreases at every event time (the event rows
    leave it), so a released set still equal to the first release at a
    later event time is necessarily a held (stale) release. The fraction is
    the share of the node's events at held grid points; the first grid
    point is always released and never counts as held.

    With the risk-set guards off (``min_risk_set_change <= 1``) nothing can
    be held, so the fraction is 0 by definition.
    """
    if min_risk_set_change <= 1:
        return 0.0
    agg1 = np.asarray(agg1, dtype=float)
    freq = np.asarray(freq, dtype=float)
    total = float(freq.sum())
    if total <= 0 or len(agg1) < 2:
        return 0.0
    unchanged = agg1[1:] == agg1[0]
    held_events = float(freq[1:][unchanged].sum())
    return held_events / total


def pooled_standardisation(z_sum, sum_squares, n_events: float) -> tuple[np.ndarray, np.ndarray]:
    """
    The pooled covariate centre and scale over the event cases.

    ``centre = z_sum / n_events`` and ``scale = sqrt(sum_squares / n_events -
    centre^2)``, both rounded to two significant figures so the nodes learn
    less about the pooled event-case covariate distribution. A covariate
    with no spread over the events gets scale 1 (with a warning); the fit is
    unaffected either way because the partial likelihood is invariant to the
    shared affine transform.
    """
    z = np.asarray(z_sum, dtype=float)
    squares = np.asarray(sum_squares, dtype=float)
    centre = z / n_events
    variance = squares / n_events - centre**2
    scale = np.sqrt(np.maximum(variance, 0.0))
    if np.any(scale <= 0):
        warn("A covariate has no spread over the event cases; using scale 1 for it.")
        scale = np.where(scale > 0, scale, 1.0)
    return round_sig(centre), round_sig(scale)


def survival_curves(
    freqs,
    summed_agg1,
    grid_times,
    beta,
    centre: np.ndarray,
    scale: np.ndarray,
    expl_vars: List[str],
    covariate_profiles: Optional[List[Dict[str, float]]],
    curve_time_sig_figs: Optional[int] = None,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Breslow baseline cumulative hazard and survival curves, all centrally.

    ``H0(t) = sum_{t_j <= t} d_j / S0(t_j)`` with ``d_j`` the pooled event
    count per grid time and ``S0(t_j)`` the risk-set sum at the reported
    (standardised-space) beta; the baseline profile is the covariate centre
    (``x* = 0``). Each requested profile gets
    ``S(t | x) = S0(t) ** exp(beta . (x - centre) / scale)``; profile
    variables left out are taken at the centre. Everything is computed from
    quantities that are already aggregated: no new data leaves a node.

    The curve time keys are rounded to ``curve_time_sig_figs`` significant
    figures when it is given; the event-time grid itself (and the hazard
    values) are unaffected. Rounding can merge nearby times into one key;
    the dicts are built in ascending time order, so a merged key carries
    the value at the largest original time. ``None`` (the default) reports
    the exact grid times.

    Returns
    -------
    tuple[dict, dict]
        The baseline cumulative hazard and the survival curves, both JSON
        objects keyed by (stringified) grid time.
    """
    d = np.asarray(freqs, dtype=float)
    s0 = np.asarray(summed_agg1, dtype=float)
    h0_steps = np.where(d > 0, d / s0, 0.0)
    h0 = np.cumsum(h0_steps)

    grid_arr = np.asarray(grid_times, dtype=float)
    if curve_time_sig_figs is None:
        key_times = grid_arr
    else:
        key_times = round_sig(grid_arr, curve_time_sig_figs)

    baseline_cumulative_hazard = {str(t): float(h) for t, h in zip(key_times, h0)}
    curves: Dict[str, Dict[str, float]] = {"0": {str(t): float(np.exp(-h)) for t, h in zip(key_times, h0)}}

    for profile_index, profile in enumerate(covariate_profiles or [], start=1):
        x_star = np.array(
            [(float(profile.get(var, centre[j])) - centre[j]) / scale[j] for j, var in enumerate(expl_vars)]
        )
        profile_exponent = float(np.exp(np.dot(beta, x_star)))
        curves[str(profile_index)] = {str(t): float(np.exp(-h * profile_exponent)) for t, h in zip(key_times, h0)}

    return baseline_cumulative_hazard, curves


def round_sig(values, digits: int = 2) -> np.ndarray:
    """Round to ``digits`` significant figures.

    The pooled covariate centre and scale are shared with the nodes;
    rounding to two significant figures means the nodes learn less about
    the pooled event-case covariate distribution.
    """
    arr = np.asarray(values, dtype=float)
    rounded = np.zeros_like(arr)
    nonzero = arr != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.floor(np.log10(np.abs(arr[nonzero])))
        factor = 10.0 ** (digits - 1 - exponent)
        rounded[nonzero] = np.round(arr[nonzero] * factor) / factor
    return rounded


def back_transform_results(model: dict, scale) -> dict:
    """Map model statistics from the standardised space back to the original
    covariates.

    ``perform_iteration`` computes its aggregates on ``(x - centre) / scale``;
    the fitted coefficients in that space map back as ``beta / scale``, the
    standard errors as ``SE / scale`` and the covariance as
    ``C / (scale scale^T)``. The Z statistic and p-values are invariant.
    """
    scale = np.asarray(scale, dtype=float)
    coef = np.asarray(model["results_data"]["Coef"], dtype=float) / scale
    se = np.asarray(model["results_data"]["SE"], dtype=float) / scale
    covariance = np.asarray(model["covariance"], dtype=float) / np.outer(scale, scale)

    model["results_data"] = {
        "Coef": coef,
        "Exp(coef)": np.exp(coef),
        "SE": se,
        "lower_CI": np.exp(coef - 1.96 * se),
        "upper_CI": np.exp(coef + 1.96 * se),
        # Z and p-values are invariant under the affine transform
        "Z": model["results_data"]["Z"],
        "p-value": model["results_data"]["p-value"],
    }
    model["covariance"] = covariance
    model["standard_errors"] = se
    return model


def format_results_dataframe(results_data: Dict[str, Any], expl_vars: List[str]) -> pd.DataFrame:
    """
    Format the results data into a properly indexed DataFrame.

    Parameters
    ----------
    results_data : Dict[str, Any]
        Dictionary containing result columns.
    expl_vars : List[str]
        List of explanatory variable names.

    Returns
    -------
    pd.DataFrame
        Formatted results DataFrame with variable names as index.
    """
    results = pd.DataFrame(results_data)
    results["Var"] = expl_vars
    results = results.set_index("Var")
    return results
