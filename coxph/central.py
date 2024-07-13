"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""
import time
import numpy as np
import json
from scipy.stats import norm
import pandas as pd
import math
from scipy.linalg import solve

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient


@algorithm_client
def central(
        client: AlgorithmClient, time_col, outcome_col, expl_vars, organization_ids, sensitivity, epsilon,
        baseline_hf=True, binning=True, bin_type="Fixed", min_count=2, differential_privacy=True):
    """
    This function is the central part of the algorithm. It performs the main computation and coordination tasks.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    time_col (str): The name of the column in the DataFrame that contains the time data.
    outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    expl_vars (list): A list of explanatory variables to be used in the computation.
    organization_ids (list): A list of organization IDs that participate in the collaboration.

    Returns:
    pandas.DataFrame: A DataFrame containing the results of the computation.
    """

    # Collect all organization that participate in this collaboration unless specified
    if isinstance(organization_ids, list) is False:
        organisations = client.organization.list()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = organization_ids

    info(f'sending task to organizations {ids}')

    n_covs = len(expl_vars)
    epochs = 10

    if binning:

        # Step 1: Get sample size from each node
        info("Getting sample sizes from all nodes")
        input_ = {
            "method": "get_sample_size"
        }

        # create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Sample size",
            description="Getting sample sizes from all nodes to calculate bin size using Sturges' rule"
        )

        # wait for node to return results of the subtask.
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

        # create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Get local bin edges",
            description="Getting local bin edges from all nodes"
        )

        # wait for node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        # Combine bin edges from all nodes and deduplicate
        bin_edges_list = [output['bin_edges'] for output in results]
        combined_bin_edges = np.unique(np.concatenate(bin_edges_list))

        if bin_type == "Quantile":
            global_bin_edges = np.round((np.quantile(combined_bin_edges, np.linspace(0, 1, bin_size + 1))), 0).tolist()

        elif bin_type == "Fixed":
            # Calculate global min and max for re-binning
            global_min = min(combined_bin_edges)
            global_max = max(combined_bin_edges)

            # Recalculate bin edges with a common set of bin edges
            global_bin_edges = [round(edge, 0) for edge in (np.linspace(global_min, global_max, bin_size + 1)).tolist()]

        else:
            info("Unsupported bin type encountered. Exiting the algorithm.")
            return {"error": "Unsupported bin type encountered. Exiting the algorithm."}

        # Define input parameters for a subtask - get unique event times
        info("Defining input parameters for subtask - get unique event times")
        input_ = {
            "method": "get_binned_unique_event_times",
            "kwargs": {
                "time_col": time_col,
                "outcome_col": outcome_col,
                'bin_edges': global_bin_edges,
                'bin_type': bin_type,
                'min_count': min_count
            },
        }

        # create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Unique event times",
            description="Getting unique event times and their counts"
        )

        # wait for node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

    elif not binning:
        # Define input parameters for a subtask - to get all unique event times if binning is not enabled
        info("Defining input parameters for subtask - get unique event times")
        input_ = {
            "method": "get_unique_event_times",
            "kwargs": {
                "time_col": time_col,
                "outcome_col": outcome_col,
                'differential_privacy': differential_privacy,
                'sensitivity': sensitivity,
                'epsilon': epsilon
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

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Summed Z statistic",
        description="Computing the summed Z statistic"
    )

    # wait for node to return results of the subtask.
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
                'sensitivity': sensitivity,
                'epsilon': epsilon
            }
        }

        # De-serialise beta again
        beta = np.array(beta)

        # create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Start iteration",
            description="Iterating to find the optimal beta"
        )

        # wait for node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        summed_agg1 = 0
        summed_agg2 = 0
        summed_agg3 = 0

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
        np.array([np.around(beta, 5), np.around(np.exp(beta), 5), np.around(np.array(SErrors), 5)]).T,
        columns=["Coef", "Exp(coef)", "SE"])
    results['Var'] = expl_vars
    results["lower_CI"] = np.around(np.exp(results["Coef"] - 1.96 * results["SE"]), 5)
    results["upper_CI"] = np.around(np.exp(results["Coef"] + 1.96 * results["SE"]), 5)
    results["Z"] = zvalues
    results["p-value"] = pvalues
    results = results.set_index("Var")

    if baseline_hf:
        # Compute the baseline hazard function
        cumulative_hazard = compute_baseline_hazard(time_col, aggregated_time_events, unique_time_events, summed_agg1)

        return {"cumulative_hazard": cumulative_hazard.to_dict(),
                "coxph_results": results.to_dict()
        }

    return {"coxph_results": results.to_dict()}


def compute_baseline_hazard(time_col, aggregated_time_events, unique_time_events, summed_agg1):
    baseline_hazard = []
    cumulative_baseline_hazard = []
    for t in range(len(unique_time_events)):
        # Compute the baseline hazard at each unique event time
        h0_t = aggregated_time_events.loc[aggregated_time_events[time_col] == unique_time_events[t], 'freq'].values[0] / summed_agg1[t]
        baseline_hazard.append(h0_t)

        # Compute the cumulative baseline hazard at each unique event time
        H0_t = np.sum(baseline_hazard)
        cumulative_baseline_hazard.append({'time': unique_time_events[t], 'cumulative_hazard': H0_t})

    cumulative_baseline_hazard_df = pd.DataFrame(cumulative_baseline_hazard)

    return cumulative_baseline_hazard_df


def compute_derivatives(summed_agg1, summed_agg2, summed_agg3, aggregated_time_events, z_sum):
    """
    This function computes the primary and secondary derivatives needed for the central algorithm.

    Parameters:
    summed_agg1 (numpy.ndarray): The aggregated sum of the first set of values.
    summed_agg2 (numpy.ndarray): The aggregated sum of the second set of values.
    summed_agg3 (numpy.ndarray): The aggregated sum of the third set of values.
    aggregated_time_events (pandas.DataFrame): The DataFrame containing the frequency of unique event times and their counts.
    z_sum (float): The summed Z statistic.

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
