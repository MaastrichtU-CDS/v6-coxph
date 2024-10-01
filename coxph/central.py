"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import solve
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient


@algorithm_client
def central(
        client: AlgorithmClient, time_col, outcome_col, expl_vars, organization_ids, sensitivity, epsilon,
        baseline_hf=True, binning=True, bin_type="Fixed", differential_privacy=True,
        privacy_target="predictors"):
    """
    This function is the central part of the algorithm. It performs the main computation and coordination tasks.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    time_col (str): The name of the column in the DataFrame that contains the time data.
    outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    expl_vars (list): A list of explanatory variables to be used in the computation.
    organization_ids (list): A list of organization IDs that participate in the collaboration.
    sensitivity (float): The sensitivity of the Cox model coefficients for differential privacy.
    epsilon (float): The desired epsilon value for differential privacy.
    baseline_hf (bool): A boolean flag to include the cumulative baseline hazard function in the results.
    binning (bool): A boolean flag to enable binning of event times for added privacy.
    bin_type (str): The type of binning to use for event times ("Fixed" or "Quantile").
    differential_privacy (bool): A boolean flag to enable differential privacy on the aggregates.
    privacy_target (str): The target of the differential privacy ("predictors" or "aggregates").

    Returns:
    pandas.DataFrame: A DataFrame containing the results of the computation.
    """

    # Collect all organization that participate in this collaboration unless specified
    if isinstance(organization_ids, list) is False:
        organisations = client.organization.list()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = organization_ids

    info(f'Sending task to organizations {ids}')

    n_covs = len(expl_vars)
    epochs = 10

    # Create a list to store the IDs of organizations that do not meet privacy guards
    excluded_ids = []

    # Define input parameters for a subtask - sample size thresholding
    info("Defining input parameters for subtask - sample size thresholding")
    input_ = {
        "method": "sample_size_thresholding",
        "kwargs": {
            "time_col": time_col,
            "outcome_col": outcome_col
        },
    }

    n_loops = 0
    n_threshold_met = False
    while not n_threshold_met:
        # This list represents the organizations that will be excluded in the following loop
        _excluded_ids = []
        if n_loops > 2:
            error("Sample size violations should be eliminated yet criteria are not met. Exiting")
            raise ValueError("Sample size violations should be eliminated yet criteria are not. Exiting")

        n_loops += 1
        # Create a subtask for all selected organizations in the collaboration.
        info("Creating subtask for all selected organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Unique event times",
            description="Getting unique event times and their counts"
        )

        # Wait for the node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        for output in results:

            # Exclude organizations that do not meet the N-threshold
            if "N-Threshold not met" in output:
                warn(f"Insufficient samples for organization {output['N-Threshold not met']}. "
                     f"Excluding organization from analysis.")
                ids.remove(output["N-Threshold not met"])
                excluded_ids.append(output["N-Threshold not met"])
                _excluded_ids.append(output["N-Threshold not met"])
                continue

        if len(_excluded_ids) == 0:
            n_threshold_met = True
        elif len(ids) == 0:
            warn("No organizations meet the minimal sample size threshold, returning NaN.")
            return {"excluded_organizations": excluded_ids, "coxph_result": np.nan}

    if binning:
        results = handle_binning(client, time_col=time_col, outcome_col=outcome_col, bin_type=bin_type,
                                 differential_privacy=differential_privacy, sensitivity=sensitivity, epsilon=epsilon,
                                 ids=ids)

    elif not binning:
        # Define input parameters for a subtask - to get all unique event times if binning is not enabled
        info("Defining input parameters for subtask - get unique event times")
        input_ = {
            "method": "get_unique_event_times",
            "kwargs": {
                "time_col": time_col,
                "outcome_col": outcome_col
            },
        }

        info("Creating subtask for all selected organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Unique event times",
            description="Getting unique event times and their counts"
        )

        # Wait for the node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

    unique_time_events = []
    for output in results:
        output = pd.DataFrame.from_dict(output["times"])
        unique_time_events.append(output)

    aggregated_time_events = pd.concat(unique_time_events)
    aggregated_time_events = aggregated_time_events.groupby(time_col, as_index=False).sum()
    # Get the list of unique_time_events
    unique_time_events = aggregated_time_events[time_col].tolist()

    # Define input parameters for a subtask - compute_summed_z
    info("Defining input parameters for subtask - compute summed z")
    input_ = {
        "method": "compute_summed_z",
        "kwargs": {
            "outcome_col": outcome_col,
            "expl_vars": expl_vars
        }
    }

    # Create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Summed Z statistic",
        description="Computing the summed Z statistic"
    )

    # Wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    z_sum = 0
    for output in results:
        z_sum += pd.Series(output["sum"])

    beta = np.zeros(n_covs)

    for epoch in range(epochs):

        # JSON-serialize beta for Vantage6
        beta = beta.tolist()

        # Define input parameters for a subtask - perform iterations
        info("Defining input parameters for subtask - perform iterations")
        input_ = {
            "method": "perform_iteration",
            "kwargs": {
                'time_col': time_col,
                "expl_vars": expl_vars,
                'beta': beta,
                'unique_time_events': unique_time_events,
                'differential_privacy': differential_privacy,
                'privacy_target': privacy_target,
                'sensitivity': sensitivity,
                'epsilon': epsilon
            }
        }

        # De-serialise beta again
        beta = np.array(beta)

        # Create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Start iteration",
            description="Iterating to find the optimal beta"
        )

        # Wait for node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        summed_agg1 = 0
        summed_agg2 = 0
        summed_agg3 = 0
        snr = 0

        for output in results:
            summed_agg1 += np.array(output['agg1'])
            summed_agg2 += np.array(pd.DataFrame.from_dict(output['agg2']))
            summed_agg3 += np.array([np.array(lst) for lst in output['agg3']])

        primary_derivative, secondary_derivative = compute_derivatives(summed_agg1, summed_agg2, summed_agg3,
                                                                       aggregated_time_events,
                                                                       z_sum)

        beta_old = np.array(beta)
        beta = beta_old - solve(secondary_derivative, primary_derivative)
        delta = float(max(abs(beta - beta_old)))

        if math.isnan(delta):
            info("Delta has turned into a NaN?")
            break

        if delta <= 0.000001:
            info("Betas have settled! Finished iterating!")
            break

    # Computing the standard errors
    SErrors = []
    fisher = np.linalg.inv(-secondary_derivative)
    for k in range(fisher.shape[0]):
        SErrors.append(np.sqrt(fisher[k, k]))

    # Calculating P and Z values
    zvalues = (np.exp(beta) - 1) / np.array(SErrors)
    pvalues = 2 * norm.cdf(-abs(zvalues))

    # 95%CI = beta +- 1.96 * SE
    results = pd.DataFrame(
        np.array([np.around(beta, 5), np.around(np.exp(beta), 5),
                  np.around(np.array(SErrors), 5)]).T,
        columns=["Coef", "Exp(coef)", "SE"])
    results['Var'] = expl_vars
    results["lower_CI"] = np.around(np.exp(results["Coef"] - 1.96 * results["SE"]), 5)
    results["upper_CI"] = np.around(np.exp(results["Coef"] + 1.96 * results["SE"]), 5)
    results["Z"] = zvalues
    results["p-value"] = pvalues
    results = results[['Var'] + [col for col in results.columns if col != 'Var']]

    if baseline_hf:
        # Compute the cumulative baseline hazard and survival function
        survival_function, cumulative_hazard = compute_baseline_hazard(time_col,
                                                                       aggregated_time_events,
                                                                       unique_time_events, summed_agg1)

        return {"included_organizations": ids, "excluded_organizations": excluded_ids,
                # TODO remove this workaround as soon as .to_json() is supported for line plots
                "cumulative_baseline_hazard": cumulative_hazard.to_dict(),
                "baseline_survival_function": survival_function.to_dict(),
                # "cumulative_baseline_hazard": cumulative_hazard.to_json(),
                # "baseline_survival_function": survival_function.to_json(),
                "coxph_results": results.to_json()
                }

    return {"included_organizations": ids, "excluded_organizations": excluded_ids, "coxph_results": results.to_json()}


@algorithm_client
def handle_binning(client: AlgorithmClient, time_col, outcome_col, bin_type,
                   differential_privacy, sensitivity, epsilon, ids):
    """
    Perform binning of event times.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    time_col (str): The name of the column in the DataFrame that contains the time data.
    outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    differential_privacy (bool): A boolean flag to enable differential privacy.
    sensitivity (float): The sensitivity of the query/function for differential privacy.
    epsilon (float): The privacy budget for differential privacy.
    ids (list): A list of organization IDs that participate in the collaboration.
    bin_type (str): The type of binning to use for event times ("Fixed" or "Quantile").

    Returns:
    dict: A dictionary containing the results of the binning process.
    """
    info("Binning is enabled. Proceeding with binning.")

    # Step 1: Get sample size from each node
    info("Getting sample sizes from all nodes")
    input_ = {
        "method": "get_sample_size"
    }

    # Create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Sample size",
        description="Getting sample sizes from all nodes to calculate bin size using Sturges' rule"
    )

    # Wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    total_sample_size = sum([output["sample_size"] for output in results])
    bin_size = int(math.ceil(math.log2(total_sample_size) + 1))  # Sturges' rule

    # Step 2: Get local bin edges from each node
    info("Getting local bin edges from all nodes")
    input_ = {
        "method": "get_local_bin_edges",
        "kwargs": {
            "time_col": time_col,
            "outcome_col": outcome_col,
            'bin_size': bin_size,
            'bin_type': bin_type,
            'differential_privacy': differential_privacy,
            'sensitivity': sensitivity,
            'epsilon': epsilon
        },
    }

    # Create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Get local bin edges",
        description="Getting local bin edges from all nodes"
    )

    # Wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    # Combine bin edges from all nodes and deduplicate
    bin_edges_list = [output['bin_edges'] for output in results]
    combined_bin_edges = np.unique(np.concatenate(bin_edges_list))

    if bin_type == "Quantile":
        global_bin_edges = np.round((np.quantile(combined_bin_edges,
                                                 np.linspace(0, 1, bin_size + 1))),
                                    0).tolist()

    elif bin_type == "Fixed":
        # Calculate global min and max for re-binning
        global_min = min(combined_bin_edges)
        global_max = max(combined_bin_edges)

        # Recalculate bin edges with a common set of bin edges
        global_bin_edges = [round(edge, 0) for edge in (np.linspace(global_min, global_max, bin_size + 1)).tolist()]

    else:
        error("Unsupported bin type encountered. Exiting the algorithm.")
        return {"error": "Unsupported bin type encountered. Exiting the algorithm."}

    # Define input parameters for a subtask - get unique event times
    info("Defining input parameters for subtask - get unique event times")
    input_ = {
        "method": "get_binned_unique_event_times",
        "kwargs": {
            "time_col": time_col,
            "outcome_col": outcome_col,
            'bin_edges': global_bin_edges,
            'bin_type': bin_type
        },
    }

    # Create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Unique event times",
        description="Getting event times and their counts based on global bin edges"
    )

    # Wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")
    return results


def compute_baseline_hazard(time_col, aggregated_time_events, unique_time_events, summed_agg1):
    """
     Compute the cumulative baseline hazard and survival function.

     Parameters:
     time_col (str): The name of the column in the DataFrame that contains the time data.
     aggregated_time_events (pandas.DataFrame): DataFrame containing the frequency of unique event times and their counts.
     unique_time_events (list): A list of unique event times.
     summed_agg1 (numpy.ndarray): The aggregated sum of the first set of values.

     Returns:
     tuple: A tuple containing:
         - pandas.DataFrame: DataFrame containing the survival function, with columns ['time', 'survival'].
         - pandas.DataFrame: DataFrame containing the cumulative baseline hazard, with columns ['time', 'hazard'].
     """
    baseline_hazard = []
    cumulative_baseline_hazard = []
    for t in range(len(unique_time_events)):
        # Compute the hazard at each unique event time using the Breslow estimator
        hazard = aggregated_time_events.loc[aggregated_time_events[time_col] ==
                                            unique_time_events[t], 'freq'].values[0] / summed_agg1[t]
        baseline_hazard.append(hazard)

        # Compute the cumulative baseline hazard at each unique event time
        H0_t = np.sum(baseline_hazard)
        cumulative_baseline_hazard.append({'time': unique_time_events[t], 'hazard': H0_t})

    cumulative_baseline_hazard_df = pd.DataFrame(cumulative_baseline_hazard)
    baseline_survival_function_df = calculate_survival_function(cumulative_baseline_hazard_df)

    return baseline_survival_function_df, cumulative_baseline_hazard_df


def calculate_survival_function(cumulative_baseline_hazard_df):
    """
    Calculate the survival function from the cumulative baseline hazard.

    Parameters:
    cumulative_baseline_hazard_df (pandas.DataFrame): DataFrame containing the cumulative baseline hazard,
                                           with columns ['time', 'hazard'].

    Returns:
    pandas.DataFrame: DataFrame containing the survival function, with columns ['time', 'survival'].
    """

    # Create a copy of the DataFrame to store the survival function
    baseline_survival_function = cumulative_baseline_hazard_df.copy()

    # Calculate the survival function using the formula S(t) = exp(-H(t))
    baseline_survival_function['survival'] = np.exp(-cumulative_baseline_hazard_df['hazard'])

    # Select only the necessary columns to return
    baseline_survival_function_df = baseline_survival_function[['time', 'survival']]

    return baseline_survival_function_df


def compute_derivatives(summed_agg1, summed_agg2, summed_agg3, aggregated_time_events, z_sum):
    """
    This function computes the primary and secondary derivatives needed for the central algorithm.

    Returns:
    tuple: A tuple containing the primary and secondary derivatives.

    """

    tot_p1 = 0
    tot_p2 = 0

    # Iterate over each row in the DataFrame
    for index, row in aggregated_time_events.iterrows():
        # Compute the primary derivative component
        s1 = row['freq'] * (summed_agg2[index] / summed_agg1[index])

        # Compute the first part of the secondary derivative component
        first_part = (summed_agg3[index] / summed_agg1[index])

        # Compute the second part of the secondary derivative component
        # The numerator is the outer product of agg2
        numerator = np.outer(summed_agg2[index], summed_agg2[index])
        denominator = summed_agg1[index] * summed_agg1[index]
        second_part = numerator / denominator

        s2 = row['freq'] * (first_part - second_part)

        tot_p1 += s1
        tot_p2 += s2

    # Compute the primary and secondary derivatives
    primary_derivative = z_sum - tot_p1
    secondary_derivative = -tot_p2

    return primary_derivative, secondary_derivative
