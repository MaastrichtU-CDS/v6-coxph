"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import numpy as np
import pandas as pd
import json

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.client import AlgorithmClient


@data(1)
@algorithm_client
def get_unique_event_times(client: AlgorithmClient, df: pd.DataFrame, time_col, outcome_col):
    """
    This function retrieves unique event times from the provided DataFrame.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    df (pandas.DataFrame): The DataFrame containing the data.
    time_col (str): The name of the column in the DataFrame that contains the time data.
    outcome_col (str): The name of the column in the DataFrame that contains the outcome data.

    Returns:
    dict: A dictionary containing a DataFrame of unique event times.
    """
    info("Computing unique event times")
    times = df[df[outcome_col] == 1].groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times['freq'] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    return {'times': times.to_dict()}


@data(1)
@algorithm_client
def compute_summed_z(client: AlgorithmClient, df: pd.DataFrame, outcome_col, expl_vars):
    """
    This function computes the sum of the specified explanatory variables for the outcome events.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    df (pandas.DataFrame): The DataFrame containing the data.
    outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    expl_vars (list): A list of explanatory variables to be used in the computation.

    Returns:
    dict: A dictionary containing the sum of the explanatory variables for the outcome events.
    """
    info("Computing summed z statistics")
    z_sum = (df[df[outcome_col] == 1][expl_vars].sum().to_dict())
    return {'sum': z_sum}


@data(1)
@algorithm_client
def perform_iteration(client: AlgorithmClient, df: pd.DataFrame, time_col, expl_vars, beta, unique_time_events):
    """
    This function performs an iteration of the algorithm, computing the necessary aggregates.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    df (pandas.DataFrame): The DataFrame containing the data.
    time_col (str): The name of the column in the DataFrame that contains the time data.
    expl_vars (list): A list of explanatory variables to be used in the computation.
    beta (numpy.ndarray): The current estimate of the beta coefficients.
    unique_time_events (list): A list of unique time events.

    Returns:
    dict: A dictionary containing the aggregates computed during the iteration.
    """
    info("Computing aggregates for the derivation of the partial likelihood")
    # Deserialize beta values
    beta = np.array(beta)
    num_unique_time_events = len(unique_time_events)
    num_explanatory_vars = len(expl_vars)

    agg1 = []
    agg2 = []
    agg3 = []

    for i in range(num_unique_time_events):
        R_i = df[df[time_col] >= unique_time_events[i]][expl_vars]
        # Check if R_i is empty
        if not R_i.empty:
            ebz = np.exp(np.dot(np.array(R_i), beta))
            agg1.append(sum(ebz))
            func = lambda x: np.asarray(x) * np.asarray(ebz)
            z_ebz = R_i.apply(func)
            agg2.append(z_ebz.sum())

            summed = np.zeros((num_explanatory_vars, num_explanatory_vars))
            for j in range(len(R_i)):
                summed = summed + np.outer(np.array(z_ebz)[j], np.array(R_i)[j].T)
            agg3.append(summed)

        else:
            agg1.append(0)
            agg2.append(pd.Series(np.zeros(num_explanatory_vars), index=expl_vars))
            agg3.append(np.zeros((num_explanatory_vars, num_explanatory_vars)))

    # JSON-serialize the results
    agg2 = pd.DataFrame(agg2).to_dict()
    agg3 = [array.tolist() for array in agg3]

    return {'agg1': agg1,
            'agg2': agg2,
            'agg3': agg3}
