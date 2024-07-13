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

from scipy.stats import laplace
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client


# Function to add Laplace noise for continuous data
def add_laplace_noise(data, sensitivity, epsilon):
    """
    Adds Laplace noise to the data for differential privacy.

    This function generates Laplace noise based on the provided sensitivity and epsilon (privacy budget)
    parameters and adds it to the input data. The scale of the Laplace distribution is determined by the
    sensitivity divided by epsilon. This method is typically used to ensure differential privacy of the
    numerical data by making it harder to infer information about any individual entry in the dataset.

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
    Applies the randomized response technique to boolean data for differential privacy.

    This function modifies the input boolean data by randomly flipping each value based on a probability
    determined by the privacy parameter epsilon. The probability ensures that the true value is kept with
    a certain likelihood, and flipped otherwise, providing a mechanism to preserve privacy while still
    allowing for accurate aggregate statistics.

    Parameters:
    - data (pd.Series): The original boolean data to which the randomized response will be applied.
    - epsilon (float): The privacy budget parameter controlling the trade-off between privacy and accuracy.
                       A higher epsilon value results in less privacy (and more accuracy), and vice versa.

    Returns:
    - pd.Series: The modified data after applying the randomized response technique, with integer type.
    """

    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # Probability of keeping the true value
    data = data.apply(lambda x: x if random.random() < p else 1 - x)
    return data.astype(int)


# Function to check data type and apply appropriate privacy method
def privatize_data(data, sensitivity, epsilon):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if data.dtype == bool:
            return apply_randomized_response(data, epsilon)
        else:
            return add_laplace_noise(data, sensitivity, epsilon)
    elif isinstance(data, list):
        data = np.array(data)
        noisy_data = add_laplace_noise(data, sensitivity, epsilon)
        # Convert numpy array back to list
        noisy_data = np.round(noisy_data, 2).tolist()
        return noisy_data
    elif isinstance(data, (int, float, np.float64)):
        # Check if the value is boolean/categorical
        if data in [0, 1]:
            return apply_randomized_response(pd.Series([data]), epsilon).iloc[0]
        else:
            return add_laplace_noise(np.array([data]), sensitivity, epsilon)[0]
    else:
        raise ValueError("Unsupported data type for privatization")

@data(1)
@algorithm_client
def get_unique_event_times(client: AlgorithmClient, df: pd.DataFrame, time_col, outcome_col,
                           differential_privacy, sensitivity, epsilon):
    """
    Retrieves unique event times from the provided DataFrame, applying differential privacy if required.

    This function computes the unique event times based on the specified 'time_col' and 'outcome_col'. It groups
    the DataFrame by 'time_col' to count occurrences of each unique time, where the outcome is 1. If differential
    privacy is enabled, it applies a privacy-preserving mechanism to the time data before returning the results.

    Parameters:
    - client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    - df (pd.DataFrame): The DataFrame containing the data.
    - time_col (str): The name of the column in the DataFrame that contains the time data.
    - outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    - differential_privacy (bool): Flag indicating whether differential privacy should be applied.
    - sensitivity (float): The sensitivity parameter for differential privacy. Used only if differential_privacy is True.
    - epsilon (float): The epsilon parameter for differential privacy. Used only if differential_privacy is True.

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
    if differential_privacy:
        times[time_col] = privatize_data(times[time_col], sensitivity, epsilon)
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
def get_local_bin_edges(client: AlgorithmClient, df: pd.DataFrame, time_col, outcome_col, bin_size,
                        bin_type, differential_privacy, sensitivity, epsilon):
    """
    Calculates the local bin edges for event times in the provided DataFrame, optionally applying differential privacy.

    This function determines the bin edges for event times based on the specified binning strategy ('Quantile' or 'Fixed').
    For 'Quantile' binning, quantiles are used to determine bin edges, ensuring an even distribution of event times across bins.
    For 'Fixed' binning, bin edges are calculated based on fixed intervals over the range of event times. If differential
    privacy is enabled, noise is added to the bin edges to protect the privacy of the data.

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
        # Calculate the quantiles and bin edges, rounding to 2 decimal places
        bin_edges = np.round(np.quantile(event_times, np.linspace(0, 1, bin_size + 1)), decimals=0)
        # Calculate event counts in each bin
        # event_counts, _ = np.histogram(event_times, bins=bin_edges)
        # Add noise to the bin edges to ensure privacy (only if differential privacy is enabled)
        if differential_privacy:
            bin_edges = privatize_data(bin_edges.tolist(), sensitivity, epsilon)
        else:
            bin_edges = bin_edges.tolist()

    elif bin_type == 'Fixed':
        # Calculate the bin edges using fixed bin sizes, rounding to 2 decimal places
        bin_edges = np.round(np.histogram_bin_edges(event_times, bins=bin_size), 0)
        # Calculate event counts in each bin
        # event_counts, _ = np.histogram(event_times, bins=bin_edges)
        # Add noise to the bin edges to ensure privacy (only if differential privacy is enabled)
        if differential_privacy:
            bin_edges = privatize_data(bin_edges.tolist(), sensitivity, epsilon)
        else:
            bin_edges = bin_edges.tolist()

    else:
        info("Unsupported bin type encountered. Exiting the algorithm.")
        return {"error": "Unsupported bin type encountered. Exiting the algorithm."}

    # Return the bin edges as a dictionary
    return {'bin_edges': bin_edges}


@data(1)
@algorithm_client
def get_binned_unique_event_times(client: AlgorithmClient, df: pd.DataFrame, time_col, outcome_col, bin_edges,
                                  bin_type, min_count):
    """
    Calculates the unique event times and their frequencies in the dataframe, placing them in specified bins.

    This function processes event times from the provided DataFrame, categorizing them into bins defined by
    'bin_edges'. It supports both 'Quantile' and 'Fixed' binning strategies. The function iteratively adjusts
    bin edges to ensure each bin meets a minimum count requirement ('min_count'), enhancing the privacy and
    reliability of the results. Optionally, differential privacy can be applied to further protect the data.

    Parameters:
    - client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    - df (pd.DataFrame): The DataFrame containing the data.
    - time_col (str): The name of the column in the DataFrame that contains the time data.
    - outcome_col (str): The name of the column in the DataFrame that contains the outcome data.
    - bin_edges (numpy.ndarray): The edges of the bins to be used in the calculation.
    - bin_type (str): The type of binning strategy ('Quantile' or 'Fixed').
    - min_count (int): The minimum number of events required in each bin.

    Returns:
    - dict: A dictionary containing the binned unique event times and their frequencies.

    Raises:
    - ValueError: If an unsupported bin type is provided.
    """

    max_iterations = 500  # Set a maximum number of iterations to prevent infinite loops
    iteration_count = 0  # Initialize the iteration count

    event_data = df[df[outcome_col] == 1]
    event_times = event_data[time_col]

    bin_edges = bin_edges.copy()  # Make a copy to avoid modifying the original bin_edges
    while True:
        iteration_count += 1
        # Digitize the times with global bin edges to get the indices of the event times in the bins
        binned_times = np.digitize(event_times, bins=bin_edges, right=True)
        # Count events in each bin
        if bin_type == 'Quantile':
            bin_labels = np.round(bin_edges, 0)  # Use the left edge of each bin as the bin label
        elif bin_type == 'Fixed':
            bin_labels = np.round(bin_edges[:-1], 0)  # Use the left edge of each bin as the bin label
        counts = np.zeros(len(bin_labels))

        for i in range(len(bin_labels)):
            counts[i] = event_data[(binned_times == (i + 1))].shape[0]

        # Check if all bins meet the min_count requirement
        if np.all(counts >= min_count) or iteration_count >= max_iterations:
            break  # All bins have sufficient counts, exit the loop

        # Adjust bin edges to ensure min_count in each bin
        for i in range(len(counts)):
            if counts[i] < min_count:
                # Find the closest non-zero bin and adjust edges accordingly
                if i > 0 and counts[i - 1] > min_count:
                    bin_edges[i] -= (bin_edges[i] - bin_edges[i - 1]) * 0.05
                if i < len(counts) - 1 and counts[i + 1] > min_count:
                    bin_edges[i + 1] += (bin_edges[i + 1] - bin_edges[i]) * 0.05

    # Log progress for debugging
    # print(f"Iteration {iteration_count}: Adjusting bin edges to meet min_count requirements")

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


# Function to pre-compute noise for each explanatory variable
def precompute_noise_for_expl_vars(expl_vars, sensitivity, epsilon, df_shape):
    noise_dict = {}
    scale = sensitivity / epsilon
    for col in expl_vars:
        # Generate noise based on the column data shape
        noise = laplace.rvs(scale=scale, size=df_shape)
        noise_dict[col] = noise
    return noise_dict


@data(1)
@algorithm_client
def perform_iteration(client: AlgorithmClient, df: pd.DataFrame, time_col, expl_vars, beta,
                      unique_time_events, differential_privacy, sensitivity, epsilon):
    """
    This function calculates the necessary aggregates for the Cox Proportional Hazards model by iterating over unique
    event times. It optionally applies differential privacy mechanisms to the explanatory variables before computing
    the aggregates. The function supports reusing precomputed noise for differential privacy across multiple calls
    to ensure consistency and efficiency.

    Parameters:
    - client (AlgorithmClient): The client instance used to interact with the vantage6 server.
    - df (pd.DataFrame): The DataFrame containing the data.
    - time_col (str): The name of the column in the DataFrame that contains the time data.
    - expl_vars (list): A list of explanatory variables to be used in the computation.
    - beta (numpy.ndarray): The current estimate of the beta coefficients.
    - unique_time_events (list): A list of unique time events.
    - differential_privacy (bool): Indicates whether differential privacy should be applied.
    - sensitivity (float): The sensitivity parameter for differential privacy. Used to calculate the scale of the Laplace noise added to the explanatory variables.
    - epsilon (float): The epsilon parameter for differential privacy. Controls the trade-off between privacy and accuracy.

    Returns:
    - dict: A dictionary containing the aggregates computed during the iteration. The dictionary includes 'agg1', 'agg2', and 'agg3' keys, corresponding to the primary and secondary derivatives needed for the Cox model.

    Note:
    - This function uses a static variable to track if it has been called before, allowing for the reuse of precomputed noise for differential privacy.
    """

    global precomputed_noise

    # Static variable to track if the function has been called
    if 'has_been_called' not in perform_iteration.__dict__:
        perform_iteration.has_been_called = True  # Set it to True on first call
        df_shape = len(df)  # Shape needed for noise generation
        precomputed_noise = precompute_noise_for_expl_vars(expl_vars, sensitivity, epsilon, df_shape)
    else:
        # Skip the loop on subsequent calls
        pass

    if differential_privacy:
        # Apply pre-computed noise to expl vars
        for col in expl_vars:
            # Ensure to use the same index as df to align the noise correctly
            df[col] = df[col] + precomputed_noise[col]

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

    # Add Laplace noise to the aggregates before returning them to the central server (optional)
    # agg1 = privatize_data(agg1, sensitivity, epsilon)
    # agg2 = privatize_data(agg2, sensitivity, epsilon)
    # agg3 = privatize_data(agg3, sensitivity, epsilon)

    # JSON-serialize the results
    agg2 = pd.DataFrame(agg2).to_dict()
    agg3 = [array for array in agg3]
    # Uncomment if not adding laplace noise
    agg3 = [array.tolist() for array in agg3]

    return {'agg1': agg1,
            'agg2': agg2,
            'agg3': agg3}
