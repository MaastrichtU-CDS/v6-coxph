"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import random

import numpy as np
import pandas as pd

from scipy.stats import laplace, truncnorm
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client


# Function to add Laplace noise for continuous data
def add_noise(data, sensitivity, epsilon):
    """
    Adds Laplace noise to the data for differential privacy.

    Parameters:
    - data (pd.Series or pd.DataFrame): The original data to which noise will be added.
    - sensitivity (float): The sensitivity of the query/function, which measures how much the query result
      changes when a single individual's data is added or removed from the dataset. It is a crucial parameter
      in differential privacy that helps to determine the amount of noise to add.
    - epsilon (float): The privacy budget, which controls the trade-off between privacy and accuracy. A smaller
      epsilon value provides stronger privacy guarantees but makes the output noisier (and thus less accurate).

    Returns:
    - pd.Series or pd.DataFrame: The data with Laplace noise added.
    """
    scale = sensitivity / epsilon
    noise = laplace.rvs(scale=scale, size=data.shape)
    return data + noise


# Function to apply randomized response for boolean data
def apply_randomized_response(data, epsilon):
    """
    Applies the randomized response technique to categorical data for differential privacy.

    Parameters:
    - data (pd.Series): The original data to which the randomized response will be applied.
    - epsilon (float): The privacy budget parameter controlling the trade-off between privacy and accuracy.

    Returns:
    - pd.Series: The modified data after applying the randomized response technique, with integer type.
    """

    probability_param = np.exp(epsilon) / (1 + np.exp(epsilon))  # Probability of keeping the true value
    data = data.apply(lambda x: x if random.random() < probability_param else 1 - x)
    return data.astype(int)


# Function to check data type and apply appropriate privacy method
def privatize_data(data, sensitivity, epsilon):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if data.dtype == bool:
            return apply_randomized_response(data, epsilon)
        else:
            return add_noise(data, sensitivity, epsilon)
    elif isinstance(data, list):
        data = np.array(data)
        noisy_data = add_noise(data, sensitivity, epsilon)
        # Convert numpy array back to list
        noisy_data = np.round(noisy_data, 2).tolist()
        return noisy_data
    elif isinstance(data, (int, float, np.float64)):
        # Check if the value is boolean/categorical
        if data in [0, 1]:
            return apply_randomized_response(pd.Series([data]), epsilon).iloc[0]
        else:
            return add_noise(np.array([data]), sensitivity, epsilon)[0]
    else:
        raise ValueError("Unsupported data type for privatization")


@data(1)
@algorithm_client
def get_unique_event_times(client: AlgorithmClient, df: pd.DataFrame, time_col, outcome_col):
    """
    Retrieves unique event times from the provided DataFrame, applying differential privacy if required.

    Parameters:
    - client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    - df (pd.DataFrame): The DataFrame containing the data.
    - time_col (str): The name of the column in the DataFrame that contains the time data.
    - outcome_col (str): The name of the column in the DataFrame that contains the outcome data.

    Returns:
    - dict: A dictionary containing a DataFrame of unique event times and their counts.

    Note:
    - This function is only called if the 'binning' parameter is set to False in the central task.
    """

    info("Computing unique event times")

    times = df[df[outcome_col] == 1].groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times['freq'] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    return {'times': times.to_dict()}


@data(1)
@algorithm_client
def get_sample_size(client: AlgorithmClient, df: pd.DataFrame):
    """
    This function calculates the sample size of the provided DataFrame.

    Parameters:
    client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    df (pandas.DataFrame): The DataFrame containing the data.
    Returns:
    dict: A dictionary containing the sample size of the DataFrame.
    """
    sample_size = len(df)
    return {
        "sample_size": sample_size
    }


@data(1)
@algorithm_client
def get_local_bin_edges(client: AlgorithmClient, df: pd.DataFrame, time_col, outcome_col,
                        bin_size, bin_type, differential_privacy, sensitivity, epsilon):
    """
    Calculates the local bin edges for event times in the provided DataFrame, optionally applying differential privacy.

    Parameters:
    - client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    - df (pd.DataFrame): The DataFrame containing the data.
    - time_col (str): The name of the column in the DataFrame that contains the time data.
    - outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    - bin_size (int): The size of the bins to be used in the calculation. Represents the number of bins for 'Quantile'
                      binning and the width of each bin for 'Fixed' binning.
    - bin_type (str): The type of binning strategy to use ('Quantile' or 'Fixed').
    - differential_privacy (bool): Indicates whether differential privacy should be applied to the bin edges.
    - sensitivity (float): The sensitivity parameter for differential privacy. Used to calculate the scale of the Laplace
                           noise added to the bin edges.
    - epsilon (float): The epsilon parameter for differential privacy. Controls the trade-off between privacy and accuracy.

    Returns:
    - dict: A dictionary containing the local bin edges for the event times. If an unsupported bin type is encountered,
            an error message is returned instead.
    """

    # Filter the DataFrame to only include rows where the outcome is 1
    event_data = df[df[outcome_col] == 1]
    # Extract the unique event times
    event_times = event_data[time_col]

    if bin_type == 'Quantile':
        bin_edges = np.quantile(event_times, np.linspace(0, 1, bin_size + 1))
    elif bin_type == 'Fixed':
        bin_edges = np.histogram_bin_edges(event_times, bins=bin_size)
    else:
        raise ValueError("Unsupported bin type")

    # Round bin edges to the nearest integer
    bin_edges = np.round(bin_edges, 0)
    # Calculate event counts in each bin
    # event_counts, _ = np.histogram(event_times, bins=bin_edges)
    # Add noise to the bin edges to ensure privacy (only if differential privacy is enabled)
    if differential_privacy:
        bin_edges = privatize_data(bin_edges.tolist(), sensitivity, epsilon)
    else:
        bin_edges = bin_edges.tolist()

    # Return the bin edges as a dictionary
    return {'bin_edges': bin_edges}


@data(1)
@algorithm_client
def get_binned_unique_event_times(client: AlgorithmClient, df: pd.DataFrame, time_col, outcome_col,
                                  bin_edges, bin_type):
    """
    Calculates the unique event times and their frequencies in the dataframe, placing them in specified bins.

    Parameters:
    - client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    - df (pd.DataFrame): The DataFrame containing the data.
    - time_col (str): The name of the column in the DataFrame that contains the time data.
    - outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    - bin_edges (numpy.ndarray): The edges of the bins to be used in the calculation.
    - bin_type (str): The type of binning strategy ('Quantile' or 'Fixed').

    Returns:
    - dict: A dictionary containing the binned unique event times and their frequencies.

    Raises:
    - ValueError: If an unsupported bin type is provided.
    """

    event_data = df[df[outcome_col] == 1]
    event_times = event_data[time_col]

    binned_times = np.digitize(event_times, bins=bin_edges, right=True)
    if bin_type == 'Quantile':
        bin_labels = np.round(bin_edges, 0)  # Use all the edges for the bin label
    elif bin_type == 'Fixed':
        bin_labels = np.round(bin_edges[:-1], 0)  # Use the left edge of each bin as the bin label
    counts = np.zeros(len(bin_labels))

    for i in range(len(bin_labels)):
        counts[i] = event_data[(binned_times == (i + 1))].shape[0]

    # Create the times DataFrame
    times_df = pd.DataFrame({
        time_col: bin_labels,
        'freq': counts
    })

    return {
        'times': times_df.to_dict()
    }


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
def perform_iteration(client: AlgorithmClient, df: pd.DataFrame, time_col, expl_vars, beta,
                      unique_time_events, differential_privacy, privacy_target, sensitivity, epsilon):
    """

    Parameters:
    - client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    - df (pd.DataFrame): The DataFrame containing the data.
    - time_col (str): The name of the column in the DataFrame that contains the time data.
    - expl_vars (list): A list of explanatory variables to be used in the computation.
    - beta (numpy.ndarray): The current estimate of the beta coefficients.
    - unique_time_events (list): A list of unique time events.
    - differential_privacy (bool): Indicates whether differential privacy should be applied.
    - privacy_target (str): The target of the differential privacy mechanism ('predictors' or 'aggregates').
    - sensitivity (float): The sensitivity parameter for differential privacy. Used to calculate the scale of the Laplace noise added to the explanatory variables.
    - epsilon (float): The epsilon parameter for differential privacy. Controls the trade-off between privacy and accuracy.

    Returns:
    - dict: A dictionary containing the aggregates computed during the iteration. The dictionary includes 'agg1', 'agg2', and 'agg3' keys, corresponding to the primary and secondary derivatives needed for the Cox model.

   """

    # Apply differential privacy to a subset of the explanatory variables if privacy enabled for predictors
    if differential_privacy and privacy_target == 'predictors':
        for col in expl_vars:
            # Randomly select 25% of the data indices
            subset_indices = df.sample(frac=0.25).index
            # Apply privatize_data to the selected subset
            noisy_subset = privatize_data(df.loc[subset_indices, col], sensitivity, epsilon)
            # Ensure noisy_subset is a Series and cast to the correct dtype
            if isinstance(noisy_subset, (pd.Series, pd.DataFrame)):
                if df[col].dtype == bool:
                    noisy_subset = noisy_subset.astype(bool)
                df.loc[subset_indices, col] = noisy_subset
            else:
                raise ValueError("privatize_data did not return a Series or DataFrame")

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

    # Add Laplace noise to the aggregates before returning them to the central server
    # (if differential privacy is enabled for aggregates)
    if differential_privacy and privacy_target == 'aggregates':
        agg1 = privatize_data(agg1, sensitivity, epsilon)
        agg2 = privatize_data(agg2, sensitivity, epsilon)
        agg3 = privatize_data(agg3, sensitivity, epsilon)

    # JSON-serialize the results
    agg2 = pd.DataFrame(agg2).to_dict()
    agg3 = [array for array in agg3]
    # # if noise not added to the aggregates, convert numpy arrays to lists
    if not differential_privacy or privacy_target != 'aggregates':
        agg3 = [array.tolist() for array in agg3]

    return {'agg1': agg1,
            'agg2': agg2,
            'agg3': agg3}
