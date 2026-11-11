"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""

from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from scipy.linalg import solve
from scipy.stats import chi2
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import AlgorithmError, UserInputError
from vantage6.algorithm.tools.util import info, warn

from .coxph_logic import (
    back_transform_results,
    compute_derivatives,
    compute_model_results,
    format_results_dataframe,
    held_event_fraction,
    partial_log_likelihood,
    pooled_standardisation,
    survival_curves as survival_curves_from_aggregates,
)
from .miscellaneous import IterationResult, SummedZResult, format_validation_error, validate_coxph_input
from .privacy_guards import (
    ENV_MAX_COVARIATES,
    ENV_MAX_GRID_POINTS,
    ENV_MAX_HELD_EVENT_FRACTION,
    ENV_TIME_BIN_WIDTH,
    WIRE_AGGREGATE_SIG_FIGS,
    load_curve_time_sig_figs,
    load_max_held_event_fraction,
    load_resource_caps,
)

# Maximum Newton-Raphson iterations. Module-level so tests can monkeypatch
# it to force non-convergence.
MAX_ITERATIONS = 20

# The held-event fraction diagnostic warns when more than this share of a
# node's events was scored against a held (unchanged) risk set.
HELD_FRACTION_WARN_THRESHOLD = 0.25

# Convergence tolerances for the Newton-Raphson loop. The primary
# tolerance is the exactness target: converged when max |step| <=
# STEP_TOLERANCE. The secondary tolerance declares convergence when the
# log-likelihood stops improving beyond the released-precision jitter
# (ll_tolerance) for two consecutive accepted evaluations while the step
# stays below STABLE_STEP_TOLERANCE — coarse aggregate rounding can stall
# the step above the primary tolerance without the fit having moved.
STEP_TOLERANCE = 1e-6
STABLE_STEP_TOLERANCE = 1e-3

_ModelT = TypeVar("_ModelT", bound=BaseModel)


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: str,
    outcome_col: str,
    expl_vars: list,
    organization_ids: Optional[list] = None,
    covariate_profiles: Optional[list] = None,
) -> dict:
    """
    Central function for the federated Cox Proportional Hazards algorithm.

    This function orchestrates the federated computation by:
    1. Validating input parameters
    2. Collecting organisation IDs
    3. Dispatching subtasks to compute summed Z statistics (and with them the
       event-time grid)
    4. Iteratively optimising beta coefficients using Newton-Raphson
    5. Computing final model statistics and returning results

    Parameters
    ----------
    client : AlgorithmClient
        The vantage6 client instance.
    time_col : str
        Name of the column containing time data.
    outcome_col : str
        Name of the column containing outcome/event data (1=event, 0=censored).
    expl_vars : list
        List of explanatory variables to include in the model: plain column
        names, or single-entry mappings of a column name to a categorical
        spec (``reference`` and ``categories``) or a scored spec (``scores``,
        optional ``missing``). Forwarded to the partials unchanged; the model
        terms they expand to size the fit.
    organization_ids : list, optional
        List of organisation IDs to include. If None, all organisations are used.
        Organisations that do not meet the sample-size threshold fail the
        whole run with ``PrivacyThresholdViolation`` — select them explicitly
        here rather than relying on automatic exclusion.
    covariate_profiles : list, optional
        Optional covariate profiles for survival curves. Each entry maps
        explanatory variable names to values: a number for a plain or scored
        variable, a category label for a categorical variable. Variables left
        out are taken at the pooled event-case mean (the baseline profile).

    Returns
    -------
    dict
        Dictionary containing model results, p-values, AIC, and warnings.
    """
    # Validate input parameters
    validated = validate_coxph_input(
        time_col=time_col,
        outcome_col=outcome_col,
        expl_vars=expl_vars,
        organization_ids=organization_ids,
        covariate_profiles=covariate_profiles,
    )

    time_col = validated.time_col
    outcome_col = validated.outcome_col
    expl_vars = validated.expl_vars
    organization_ids = validated.organization_ids
    # The model terms (one per plain or scored variable, one per
    # non-reference category) size the fit: the Hessian, the covariate cap,
    # the result table and the order of covariance/beta/centre/scale.
    model_terms = validated.model_terms

    # Resource caps bound the problem size the algorithm accepts; they are
    # configuration, read centrally. The message names the user-supplied
    # count and the configured cap only.
    caps = load_resource_caps()
    if len(model_terms) > caps.max_covariates:
        raise UserInputError(
            f"Too many explanatory variables: {len(model_terms)} model terms requested, the configured "
            f"maximum ({ENV_MAX_COVARIATES}) is {caps.max_covariates}."
        )

    # The reported curve time keys are coarsened to this many significant
    # figures (an empty COXPH_CURVE_TIME_SIG_FIGS reports exact times).
    curve_time_sig_figs = load_curve_time_sig_figs()

    # The held-event fraction abort cap (central side): above this share of
    # a node's events scored against a held (unchanged) risk set the run
    # aborts; an empty COXPH_MAX_HELD_EVENT_FRACTION disables the abort.
    max_held_event_fraction_cap = load_max_held_event_fraction()

    # Collect all organisations unless specified
    ids: list
    if organization_ids is None:
        organisations = client.organization.list()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = list(organization_ids)
        known_ids = {organisation.get("id") for organisation in client.organization.list()}
        unknown_ids = [org_id for org_id in ids if org_id not in known_ids]
        if unknown_ids:
            raise UserInputError(
                f"Unknown organisation IDs {unknown_ids}; the collaboration has "
                f"organisations {sorted(org_id for org_id in known_ids if org_id is not None)}."
            )

    info(f"Sending task to organisations {ids}")

    n_covs = len(model_terms)
    epochs = MAX_ITERATIONS

    # Subtask: compute summed Z
    info("Defining input parameters for subtask — compute summed Z")
    input_ = {
        "method": "compute_summed_z",
        "kwargs": {
            "time_col": time_col,
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
        },
    }

    info("Creating subtask for all organisations")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Summed Z statistic",
        description="Computing the summed Z statistic",
    )

    info(f"Waiting for results of task: {task.get('id')}")
    results = client.wait_for_results(task_id=task.get("id"))
    info(f"Results obtained of task: {task.get('id')}!")
    _require_all_organisations_answered(results, ids)

    z_sum = None
    sum_squares = None
    time_event_dfs = []
    node_privacy_settings = []
    node_info = {}
    for output in results:
        result = _parse_result(
            SummedZResult, output, {"expected_ids": ids, "expl_vars": model_terms, "time_col": time_col}
        )
        if z_sum is None:
            z_sum = pd.Series(result.sum)
            sum_squares = pd.Series(result.sum_squares)
        else:
            z_sum += pd.Series(result.sum)
            sum_squares += pd.Series(result.sum_squares)
        # Collect per-time event counts (NaN-consistent with z_sum)
        time_event_df = pd.DataFrame.from_dict(result.times)
        time_event_dfs.append(time_event_df)
        node_privacy_settings.append(result.privacy_settings)

        # Per-node event counts and settings, paired by organisation id for
        # the held-fraction diagnostic (results are not in dispatch order).
        node_info[result.organization_id] = (
            time_event_df[time_col].to_numpy(dtype=float),
            time_event_df["freq"].to_numpy(dtype=float),
            result.privacy_settings,
        )

    # Build aggregated_time_events from compute_summed_z results: it drops
    # NaN in time, outcome and expl_vars, so its event counts are consistent
    # with z_sum and the risk sets.
    aggregated_time_events = pd.concat(time_event_dfs)
    aggregated_time_events = aggregated_time_events.groupby(time_col, as_index=False).sum()

    unique_time_events = aggregated_time_events[time_col].tolist()

    # The event-time grid is data-derived, so the message names the configured
    # cap and the remedies, never the actual grid size.
    if len(unique_time_events) > caps.max_grid_points:
        raise UserInputError(
            f"The pooled event-time grid exceeds the configured maximum of "
            f"{caps.max_grid_points} points ({ENV_MAX_GRID_POINTS}). Enable time binning "
            f"({ENV_TIME_BIN_WIDTH}) on the nodes or select fewer organisations."
        )

    # Standardisation: the pooled covariate centre and spread over the event
    # cases, rounded to two significant figures so the nodes learn less. The
    # partial likelihood is invariant to this shared affine transform; the
    # nodes' aggregates are computed on (x - centre) / scale and the reported
    # coefficients are transformed back.
    assert z_sum is not None and sum_squares is not None
    n_events = float(aggregated_time_events["freq"].sum())
    centre, scale = pooled_standardisation(z_sum.to_numpy(dtype=float), sum_squares.to_numpy(dtype=float), n_events)
    info("Dispatching standardised covariates (centre and scale to 2 significant figures).")

    # Aggregate summary of the guards that were active on the nodes:
    # configuration, not data; no per-node breakdown. A warning is added
    # when any guard was active, because the estimates are then approximate.
    max_min_risk_set_change = max(int(s.get("min_risk_set_change") or 1) for s in node_privacy_settings)
    time_binning = any(s.get("time_bin_width") for s in node_privacy_settings)
    max_min_time_event_count = max(int(s.get("min_time_event_count") or 1) for s in node_privacy_settings)
    # The absolute risk-set floor is configuration too; it only takes effect
    # on nodes whose jump guard is on (k > 1), which the warning names.
    floors = [int(s["min_risk_set_size"]) for s in node_privacy_settings if s.get("min_risk_set_size") is not None]
    max_min_risk_set_size = max(floors) if floors else None
    guards_active = max_min_risk_set_change > 1 or time_binning or max_min_time_event_count > 1
    privacy_guards: dict = {
        "active": guards_active,
        "max_min_risk_set_change": max_min_risk_set_change,
        "time_binning": time_binning,
        "max_min_time_event_count": max_min_time_event_count,
        "max_min_risk_set_size": max_min_risk_set_size,
    }
    guards_warning = None
    if guards_active:
        floor_note = ""
        if max_min_risk_set_size is not None and max_min_risk_set_change > 1:
            floor_note = f", minimum risk-set size R={max_min_risk_set_size}"
        guards_warning = (
            "Privacy guards were active on one or more nodes "
            f"(maximum minimum risk-set change k={max_min_risk_set_change}, "
            f"time binning {'on' if time_binning else 'off'}, "
            f"minimum per-time event count m={max_min_time_event_count}"
            f"{floor_note}); "
            "the estimates are approximate."
        )
        warn(guards_warning)

    # The nodes round the aggregates they release to a configurable number
    # of significant figures and then quantise them to float32 precision for
    # the wire (always on), so the centrally computed log-likelihood carries
    # a rounding jitter of order events * 10^(1 - N) per node, where N is
    # the coarser of the configured rounding and the wire precision. Near
    # the optimum the true per-step gain falls below that jitter, so a
    # decrease smaller than the jitter floor is not a real decrease.
    jitter = sum(
        float(time_event_df["freq"].sum()) * 10.0 ** (1 - _released_sig_figs(settings))
        for time_event_df, settings in zip(time_event_dfs, node_privacy_settings)
    )
    ll_tolerance = max(1e-12, jitter)

    # The gradient in the standardised space needs the event-case covariate
    # sums in that space as well.
    z_sum_star = pd.Series((z_sum.to_numpy(dtype=float) - centre * n_events) / scale, index=z_sum.index)

    beta: np.ndarray = np.zeros(n_covs)

    # The reported beta, Hessian and summed_agg1 are always evaluated at the
    # same beta: the last *accepted* evaluation. A decrease of the
    # log-likelihood is only detectable at the next round-trip, so the
    # previous accepted state is kept and step-halving backs off toward it.
    accepted: dict | None = None  # beta / secondary / summed_agg1 / log_likelihood
    log_likelihood_null: float | None = None
    step_halvings = 0
    last_step: np.ndarray | None = None
    converged = False
    convergence_cause: str | None = None
    convergence_tolerance = STEP_TOLERANCE
    held_fraction_warning: str | None = None
    stable_streak = 0
    epoch = 0

    for epoch in range(epochs):
        info("Defining input parameters for subtask — perform iteration")
        input_ = {
            "method": "perform_iteration",
            "kwargs": {
                "time_col": time_col,
                "outcome_col": outcome_col,
                "expl_vars": expl_vars,
                "beta": beta.tolist(),
                "centre": centre.tolist(),
                "scale": scale.tolist(),
                "unique_time_events": unique_time_events,
            },
        }

        info("Creating subtask for all organisations")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Start iteration",
            description="Iterating to find the optimal beta",
        )

        info(f"Waiting for results of task: {task.get('id')}")
        results = client.wait_for_results(task_id=task.get("id"))
        info(f"Results obtained of task: {task.get('id')}!")
        _require_all_organisations_answered(results, ids)

        n_times = len(unique_time_events)
        summed_agg1 = np.zeros(n_times)
        summed_agg2 = np.zeros((n_times, n_covs))
        summed_agg3 = np.zeros((n_times, n_covs, n_covs))
        round1_agg1 = {}

        for output in results:
            iteration_result = _parse_result(
                IterationResult, output, {"expected_ids": ids, "n_times": n_times, "n_covs": n_covs}
            )
            # Structural sanity bounds on the released risk-set sizes: a
            # broken or grossly tampering node is rejected here (every
            # round — the check is O(T), and round-1-only would miss
            # later tampering).
            _validate_riskset_aggregates(iteration_result.organization_id, iteration_result.agg1)
            summed_agg1 += np.asarray(iteration_result.agg1, dtype=float)
            summed_agg2 += np.asarray(iteration_result.agg2, dtype=float)
            summed_agg3 += np.asarray(iteration_result.agg3, dtype=float)
            if epoch == 0:
                round1_agg1[iteration_result.organization_id] = np.asarray(iteration_result.agg1, dtype=float)

        if epoch == 0:
            # Round 1 runs at beta = 0, where the released risk-set sizes
            # are exact integers: a release still at the first released
            # risk set is exactly agg1[i] == agg1[0], and (the true risk
            # set strictly decreases at every event time) necessarily a
            # held (stale) one. The share of each node's events scored
            # against a held risk set diagnoses how much of the timeline
            # the guards held back; the maximum across nodes is reported
            # in the privacy guards summary.
            max_held_event_fraction = _max_held_event_fraction(round1_agg1, node_info, unique_time_events)
            privacy_guards["max_held_event_fraction"] = max_held_event_fraction
            if max_held_event_fraction_cap is not None and max_held_event_fraction > max_held_event_fraction_cap:
                # The fit is dominated by held risk sets; the estimates
                # would not be trustworthy. The message names the configured
                # cap and the measured fraction (the same statistic the
                # result reports) and the remedies.
                raise UserInputError(
                    f"The held-event fraction {max_held_event_fraction:.3f} (maximum across nodes) "
                    f"exceeds the configured cap {max_held_event_fraction_cap} "
                    f"({ENV_MAX_HELD_EVENT_FRACTION}): the risk-set guards held back too much of "
                    f"the event timeline and the fit is dominated by held risk sets. Use larger "
                    f"nodes, lower the risk-set guards by policy (COXPH_MIN_RISK_SET_CHANGE, "
                    f"COXPH_MIN_RISK_SET_SIZE), or run with the minimal privacy level "
                    f"(COXPH_PRIVACY_LEVEL=minimal) for validation."
                )
            if max_held_event_fraction > HELD_FRACTION_WARN_THRESHOLD:
                held_fraction_warning = (
                    f"The risk-set guards held back a large share of the event timeline: "
                    f"{max_held_event_fraction:.3f} of the events (maximum across nodes) were "
                    f"scored against an unchanged (held) risk set, above the warn threshold "
                    f"{HELD_FRACTION_WARN_THRESHOLD}; the estimates may be dominated by held "
                    f"risk sets."
                )
                warn(held_fraction_warning)

        # The log-likelihood at the current beta, from aggregates that are
        # already collected (no extra round-trip).
        ll_here = partial_log_likelihood(beta, z_sum_star, aggregated_time_events, summed_agg1)
        if log_likelihood_null is None and not beta.any():
            log_likelihood_null = ll_here

        if accepted is not None and ll_here < accepted["log_likelihood"] - ll_tolerance:
            # The step away from the accepted beta decreased the likelihood:
            # halve it and retry from the accepted point. The accepted state
            # is reported if the budget runs out mid-halving.
            step_halvings += 1
            warn("Newton step decreased the log-likelihood; halving the step.")
            last_step = np.asarray(last_step, dtype=float) * 0.5
            beta = accepted["beta"] - last_step
            continue

        primary_derivative, secondary_derivative = compute_derivatives(
            summed_agg1,
            summed_agg2,
            summed_agg3,
            aggregated_time_events,
            z_sum_star,
        )

        # Accept this evaluation: the reported beta, Hessian and summed_agg1
        # stay evaluated at the same beta. A log-likelihood gain within the
        # released-precision jitter means the likelihood has stopped
        # improving measurably; two consecutive such accepted evaluations
        # arm the secondary convergence test below.
        if accepted is not None:
            if abs(ll_here - accepted["log_likelihood"]) <= ll_tolerance:
                stable_streak += 1
            else:
                stable_streak = 0
        accepted = {
            "beta": beta,
            "secondary": secondary_derivative,
            "summed_agg1": summed_agg1,
            "log_likelihood": ll_here,
        }

        try:
            step = solve(secondary_derivative, primary_derivative)
        except np.linalg.LinAlgError as e:
            convergence_cause = f"Hessian is singular: {e}"
            warn(convergence_cause)
            break

        delta = float(np.max(np.abs(step)))

        if not np.isfinite(delta):
            convergence_cause = "Newton step is not finite"
            warn(convergence_cause)
            break

        if delta <= STEP_TOLERANCE:
            info("Betas have settled! Finished iterating!")
            converged = True
            break

        if stable_streak >= 2 and delta <= STABLE_STEP_TOLERANCE:
            # The log-likelihood has not improved beyond the
            # released-precision jitter for two consecutive accepted
            # evaluations while the step stays small: the fit has settled
            # at the released precision.
            info("Betas have settled at the released precision! Finished iterating!")
            converged = True
            convergence_tolerance = STABLE_STEP_TOLERANCE
            break

        if epoch == epochs - 1:
            # Do NOT apply the last step so the reported beta,
            # Hessian and summed_agg1 stay consistent.
            break

        last_step = step
        beta = beta - step

    n_iterations = epoch + 1
    central_warnings = [w for w in (guards_warning, held_fraction_warning) if w is not None]
    if accepted is None:
        raise AlgorithmError("The optimiser never accepted an evaluation; no model can be reported.")
    beta = accepted["beta"]
    secondary_derivative = accepted["secondary"]
    summed_agg1 = accepted["summed_agg1"]
    log_likelihood = accepted["log_likelihood"]
    if log_likelihood_null is None:
        log_likelihood_null = log_likelihood
    if not converged:
        msg = (
            f"Newton-Raphson did not converge in {n_iterations} iterations; "
            f"SE/p-values may be unreliable; statistics are reported at the "
            f"last evaluated beta"
        )
        if convergence_cause:
            msg += f"; cause: {convergence_cause}"
        warn(msg)
        central_warnings.append(msg)

    # Compute final model results — beta, secondary_derivative and
    # summed_agg1 are all evaluated at the reported beta (in the
    # standardised space). Back-transform the reported statistics to the
    # original covariates afterwards.
    model = compute_model_results(
        beta=beta,
        secondary_derivative=secondary_derivative,
        z_sum=z_sum_star,
        aggregated_time_events=aggregated_time_events,
        summed_agg1=summed_agg1,
        expl_vars=model_terms,
        converged=converged,
        n_iterations=n_iterations,
    )
    model = back_transform_results(model, scale)

    results_df = format_results_dataframe(model["results_data"], model_terms)

    # Baseline cumulative hazard and survival curves, computed centrally from
    # quantities that are already aggregated (see coxph_logic): no new
    # partial, no new data leaves a node. The profiles are expanded to
    # model-term values first (a categorical label becomes the 1/0 values of
    # that variable's terms), so the curves work on terms unchanged.
    baseline_cumulative_hazard, survival_curves = survival_curves_from_aggregates(
        aggregated_time_events["freq"].to_numpy(dtype=float),
        summed_agg1,
        aggregated_time_events[time_col].to_numpy(dtype=float),
        beta,
        centre,
        scale,
        model_terms,
        validated.expanded_covariate_profiles,
        curve_time_sig_figs,
    )

    lr_statistic = 2 * (log_likelihood - log_likelihood_null)
    return {
        "model": results_df.to_dict(orient="index"),
        "overall_p_value": model["overall_p_value"],
        "aic": model["aic"],
        "degrees_of_freedom": model["n_params"],
        "warnings": model["warnings"] + central_warnings,
        "converged": converged,
        "n_iterations": n_iterations,
        "convergence_tolerance": convergence_tolerance,
        "log_likelihood": log_likelihood,
        "log_likelihood_null": log_likelihood_null,
        "lr_statistic": lr_statistic,
        "lr_p_value": float(chi2.sf(lr_statistic, n_covs)),
        "n_events": int(aggregated_time_events["freq"].sum()),
        "covariance": np.asarray(model["covariance"]).tolist(),
        "algorithm_version": _algorithm_version(),
        "baseline_cumulative_hazard": baseline_cumulative_hazard,
        "survival_curves": survival_curves,
        "curve_time_sig_figs": curve_time_sig_figs,
        "privacy_guards": privacy_guards,
    }


def _algorithm_version() -> str:
    """The installed algorithm package version, for result traceability."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("v6-cox-ph")
    except PackageNotFoundError:
        return "unknown"


def _validate_riskset_aggregates(organization_id: int, agg1) -> None:
    """Structural sanity bounds on a node's released risk-set sizes.

    Defence in depth against a broken or grossly tampering node —
    explicitly **not** a poisoning defence: a careful adversary can stay
    inside the bounds. The released risk-set sizes must be finite and
    non-negative, non-increasing along the grid (risk sets only shrink,
    with holds; rounding and float32 quantisation are monotonic, so
    legitimate releases always satisfy this), and the first released
    size must be positive (the node passed the event threshold, so its
    rows are at risk at the first pooled grid point). A node legitimately
    releases 0 at pooled grid points beyond its own data (the tail of the
    pooled grid empties).

    Raises
    ------
    AlgorithmError
        Naming the organisation, on a structurally impossible sequence.
    """
    sizes = np.asarray(agg1, dtype=float)
    if sizes.size == 0 or not np.all(np.isfinite(sizes)) or np.any(sizes < 0):
        raise AlgorithmError(
            f"Organisation {organization_id}: released risk-set sizes must be non-negative and finite."
        )
    if sizes[0] <= 0:
        raise AlgorithmError(f"Organisation {organization_id}: the first released risk-set size must be positive.")
    if np.any(np.diff(sizes) > 0):
        raise AlgorithmError(
            f"Organisation {organization_id}: released risk-set sizes increase along the "
            f"grid, which no real risk-set sequence does."
        )


def _released_sig_figs(settings) -> int:
    """The precision of the aggregates a node released.

    The coarser of the node's configured aggregate rounding and the
    always-on wire quantisation (float32 precision): the coarser rounding
    dominates the jitter in the centrally computed log-likelihood.
    """
    configured = settings.get("aggregate_sig_figs")
    if configured is None:
        return WIRE_AGGREGATE_SIG_FIGS
    return min(int(configured), WIRE_AGGREGATE_SIG_FIGS)


def _max_held_event_fraction(round1_agg1: dict, node_info: dict, grid: list) -> float:
    """The maximum across nodes of the held-event fraction.

    ``round1_agg1`` holds each node's released risk-set sizes at
    ``beta = 0`` (round 1) and ``node_info`` pairs each organisation with
    its per-time event counts and privacy settings; both are keyed by
    organisation id because the results are not in dispatch order. A node's
    event counts are aligned with the pooled event-time grid (its own
    times are a subset of it) and the fraction is computed by
    ``held_event_fraction``. A node whose risk-set guards are off
    (k <= 1) cannot hold anything and contributes 0.
    """
    grid_arr = np.asarray(grid, dtype=float)
    fractions = []
    for org_id, agg1 in round1_agg1.items():
        node_times, node_freqs, settings = node_info[org_id]
        indices = np.searchsorted(grid_arr, node_times)
        freq = np.zeros(len(grid_arr))
        np.add.at(freq, indices, node_freqs)
        min_risk_set_change = int(settings.get("min_risk_set_change") or 1)
        fractions.append(held_event_fraction(agg1, freq, min_risk_set_change))
    return max(fractions) if fractions else 0.0


def _parse_result(model_cls: type[_ModelT], output: object, context: dict) -> _ModelT:
    """Validate a partial sub-task result with its Pydantic wire model.

    The order of ``wait_for_results`` is not the dispatch order (the
    organisation id list is deduplicated and the server orders freely), so
    results carry their own ``organization_id`` and errors are attributed
    by it. A result whose id is missing or unknown fails closed before
    validation, because there is no organisation to attribute a
    validation error to. The runtime context (dispatched ids, expected
    shapes) is not part of the payload and travels via ``context``; see
    the models in ``miscellaneous``.

    Raises
    ------
    AlgorithmError
        If the result is not a dict, its ``organization_id`` is not one of
        the dispatched organisations, or it fails the model's validation.
    """
    if not isinstance(output, dict):
        raise AlgorithmError(f"Expected a dict result, got {type(output).__name__}")
    org_id = output.get("organization_id")
    if org_id is None or org_id not in context.get("expected_ids", []):
        raise AlgorithmError(
            f"Result carries organization_id {org_id!r}, which is not one of " f"the dispatched organisations."
        )
    try:
        return model_cls.model_validate(output, context=context)
    except ValidationError as e:
        # The formatted message never echoes the offending input: for a
        # result model that input can be an aggregate array, and the
        # message ends up in the container log the researcher reads.
        raise AlgorithmError(f"Organisation {org_id}: {format_validation_error(e)}") from None


def _require_all_organisations_answered(results, ids: list) -> None:
    """Fail closed unless every dispatched organisation returned a result.

    ``wait_for_results`` returns the decoded results of the runs that have
    one; a node whose run failed (for example
    ``PrivacyThresholdViolation``) is simply absent from the list.
    Aggregating the survivors would silently compute a model over a subset
    of the collaboration — the run must fail instead. The message
    names the counts and the organisation ids that did not answer
    (ids are not data), never any data.

    Raises
    ------
    AlgorithmError
        When fewer results came back than organisations were dispatched.
    """
    valid = [r for r in results if isinstance(r, dict)]
    if len(valid) != len(ids):
        answered = {r.get("organization_id") for r in valid}
        missing = sorted(org_id for org_id in ids if org_id not in answered)
        raise AlgorithmError(
            f"Only {len(valid)} of {len(ids)} organisations returned a result "
            f"(no result from organisation ids {missing}); "
            f"refusing to aggregate a subset. The run fails closed."
        )
