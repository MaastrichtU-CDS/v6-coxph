"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""

import pandas as pd
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.exceptions import (
    AlgorithmError,
    PrivacyThresholdViolation,
    PrivacyViolation,
    UserInputError,
)
from vantage6.algorithm.tools.util import info

from .miscellaneous import IterationResult, SummedZResult
from .privacy_guards import (
    event_time_counts,
    guarded_risk_set_aggregates,
    prepare_node_data,
    prepare_time_column,
    quantise_wire_aggregates,
    resolve_expl_vars,
    round_released_aggregates,
    suppress_small_time_counts,
    validate_iteration_input,
)


@data(1)
@algorithm_client
def compute_summed_z(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    expl_vars: list,
) -> dict:
    """
    Compute the sum of the specified explanatory variables for the outcome events.

    Parameters
    ----------
    client : AlgorithmClient
        The client instance used to interact with the vantage6 server.
    df : pd.DataFrame
        The DataFrame containing the data.
    time_col : str
        The name of the column containing the time data.
    outcome_col : str
        The name of the column containing the outcome data.
    expl_vars : list
        The explanatory variables: plain column names, or single-entry
        mappings of a column name to a categorical or scored spec. The node
        re-validates the spec and encodes the categorical and scored
        entries into numeric model-term columns.

    Returns
    -------
    dict
        A dictionary containing the sum of the explanatory variables.
    """
    try:
        info("Computing summed Z statistics")

        covariates = resolve_expl_vars(expl_vars)
        df, settings, threshold_met = prepare_node_data(
            client, df, time_col, outcome_col, covariates, need_outcome=True
        )

        if not threshold_met:
            raise PrivacyThresholdViolation("Sample size threshold not met: refusing to share aggregates.")

        df = prepare_time_column(df, time_col, settings, outcome_col)

        events = df[df[outcome_col] == 1]
        z_sum = events[covariates.terms].sum().to_dict()
        # Sum of squares over the event cases: with z_sum this yields the pooled
        # covariate centre and spread over events without another round-trip.
        sum_squares = events[covariates.terms].pow(2).sum().to_dict()

        # Also return per-time event counts from the same NaN-dropped,
        # censored DataFrame so that central can build aggregated_time_events
        # that are consistent with z_sum and the risk sets.
        times = event_time_counts(df, time_col, outcome_col)

        # Small-cell suppression on the event-time grid: per-time counts below
        # the configured minimum are merged into the following time.
        merged_times, merged_counts = suppress_small_time_counts(
            times[time_col].to_numpy(dtype=float), times["freq"].to_numpy(), settings.min_time_event_count
        )
        times = pd.DataFrame({time_col: merged_times, "freq": merged_counts})

        # The node's privacy settings (configuration, not data) let central
        # report which guards were active during the run.
        privacy_settings = {
            "sample_size_threshold": settings.sample_size_threshold,
            "min_risk_set_change": settings.min_risk_set_change,
            "time_bin_width": settings.time_bin_width,
            "min_time_event_count": settings.min_time_event_count,
            "aggregate_sig_figs": settings.aggregate_sig_figs,
            "min_risk_set_size": settings.min_risk_set_size,
        }
        # The result dict is built from the wire model, so its keys and nesting
        # are pinned in one place (SummedZResult).
        return SummedZResult(
            organization_id=client.organization_id,
            sum=z_sum,
            sum_squares=sum_squares,
            times=times.to_dict(),
            privacy_settings=privacy_settings,
        ).model_dump()
    except (UserInputError, PrivacyThresholdViolation, PrivacyViolation, AlgorithmError):
        # Typed vantage6 errors are part of the contract; they surface as-is.
        raise
    except Exception as e:
        # Unexpected library errors must not leak their text into the
        # container log the researcher reads; the type name is enough.
        raise AlgorithmError(f"Unexpected error in compute_summed_z: {type(e).__name__}") from None


@data(1)
@algorithm_client
def perform_iteration(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    expl_vars: list,
    beta: list,
    centre: list,
    scale: list,
    unique_time_events: list,
) -> dict:
    """
    Perform an iteration of the algorithm, computing the necessary aggregates.

    The covariates are standardised on the node as ``(x - centre) / scale``
    before the risk-set aggregates are computed: the partial likelihood is
    invariant to this shared affine transform, while centring prevents ``exp``
    overflow from large offsets and scaling fixes the Hessian conditioning
    from large spreads. Central back-transforms the coefficients it reports.

    Parameters
    ----------
    client : AlgorithmClient
        The client instance used to interact with the vantage6 server.
    df : pd.DataFrame
        The DataFrame containing the data.
    time_col : str
        The name of the column containing the time data.
    outcome_col : str
        The name of the column containing the outcome data. Rows with a
        missing outcome are dropped so that tail censoring uses the same
        rows as ``compute_summed_z``.
    expl_vars : list
        The explanatory variables: plain column names, or single-entry
        mappings of a column name to a categorical or scored spec. The node
        re-validates the spec and encodes the categorical and scored
        entries into numeric model-term columns.
    beta : list
        The current estimate of the beta coefficients (standardised space),
        as it arrives on the wire (a JSON list).
    centre : list
        The pooled covariate means over the event cases.
    scale : list
        The pooled covariate standard deviations over the event cases.
    unique_time_events : list
        A list of unique time events.

    Returns
    -------
    dict
        A dictionary containing the aggregates computed during the iteration.
    """
    try:
        info("Computing aggregates for the derivation of the partial likelihood")

        covariates = resolve_expl_vars(expl_vars)
        df, settings, threshold_met = prepare_node_data(
            client, df, time_col, outcome_col, covariates, need_outcome=True
        )

        # A node that passed compute_summed_z passes this check as well (the same
        # rows are analysed); the threshold here is defence in depth.
        if not threshold_met:
            raise PrivacyThresholdViolation("Sample size threshold not met: refusing to share aggregates.")

        beta_arr, centre_arr, scale_arr, unique_time_events = validate_iteration_input(
            beta, centre, scale, unique_time_events, covariates.terms, settings
        )

        # Move the event rows' times up to their suppressed (merged) time so
        # that the risk sets match the event grid central builds from the
        # times payload; censored rows keep their time.
        df = prepare_time_column(df, time_col, settings, outcome_col, move_event_times=True)

        X_all = (df[covariates.terms].to_numpy(dtype=float) - centre_arr) / scale_arr

        # The categories of the categorical covariates (reference included):
        # their 0/1 indicator terms let a reader separate the released sums
        # into each category's members and the rest, so the risk-set guard
        # holds the same jump/floor guarantees for them as for the whole.
        restricted_groups = []
        for entry in covariates.categorical:
            if entry.name not in df.columns:
                continue
            term_columns = [f"{entry.name}[{label}]" for label in entry.categories if label != entry.reference]
            if not all(term in df.columns for term in term_columns):
                continue
            for term in term_columns:
                restricted_groups.append(df[term].to_numpy(dtype=float) == 1.0)
            restricted_groups.append((df[term_columns].to_numpy(dtype=float) == 0.0).all(axis=1))

        agg1, agg2, agg3 = guarded_risk_set_aggregates(
            df[time_col],
            unique_time_events,
            settings.min_risk_set_change,
            X_all,
            beta_arr,
            settings.min_risk_set_size,
            restricted_groups=restricted_groups,
        )

        # Round the released aggregates to the configured precision, then
        # quantise them to float32 precision for the wire (always on).
        agg1, agg2, agg3 = round_released_aggregates(agg1, agg2, agg3, settings.aggregate_sig_figs)
        agg1, agg2, agg3 = quantise_wire_aggregates(agg1, agg2, agg3)

        # The result dict is built from the wire model, so its keys and nesting
        # are pinned in one place (IterationResult).
        return IterationResult(
            organization_id=client.organization_id,
            agg1=agg1.tolist(),
            agg2=agg2.tolist(),
            agg3=agg3.tolist(),
        ).model_dump()
    except (UserInputError, PrivacyThresholdViolation, PrivacyViolation, AlgorithmError):
        # Typed vantage6 errors are part of the contract; they surface as-is.
        raise
    except Exception as e:
        # Unexpected library errors must not leak their text into the
        # container log the researcher reads; the type name is enough.
        raise AlgorithmError(f"Unexpected error in perform_iteration: {type(e).__name__}") from None
