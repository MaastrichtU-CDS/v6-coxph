Implementation
==============

Overview
--------

Central
--------
The central part is responsible for the orchestration and aggregation of the algorithm.

``central``
~~~~~~~~~~~~~~~~
The central part is responsible for the following tasks:

- Calls binning function if specified.
- Request the unique event times and their counts in each data station.
- Request the summed z-statistics for the explanatory variables (i.e. the predictors) in each data station.
- Orchestrate iterations in the data stations to retrieve intermediate model parameters.
- Pass the intermediate model parameters to the compute_derivatives function.
- Compute the aggregated model parameters.

``handle_binning``
~~~~~~~~~~~~~~~~~~
This function is called by the central part and is executed on the central aggregator.
- Performs binning of event times.

``compute_baseline_hazard``
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function is called by the central part and is executed on the central aggregator.
- Computes the baseline hazard.
- Calls the survival function.
- Returns baseline hazards and survival function.

``calculate_survival_function``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function is called by the central part and is executed on the central aggregator.
- Calculates the survival function based on the baseline hazard and the model parameters.

``compute_derivatives``
~~~~~~~~~~~~~~~~
The function is called by the central part and is executed on the central aggregator.
- Computes the primary and secondary derivatives needed to compute the maximum likelihood estimates of the model parameters.


Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``add_noise``
~~~~~~~~~~~~~
This function adds Laplace noise to the data for differential privacy.

``apply_randomized_response``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function applies the randomized response technique to categorical data for differential privacy.

``privatize_data``
~~~~~~~~~~~~~~~~~~
This function applies differential privacy techniques to the provided data based on its type.


``get_unique_event_times``
~~~~~~~~~~~~~~~~
This function retrieves unique event times and their counts from the selected database.

``get_sample_size``
~~~~~~~~~~~~~~~~~~~
This function calculates the sample size of the provided DataFrame.

``get_local_bin_edges``
~~~~~~~~~~~~~~~~~~~~~~~
Calculates the local bin edges for event times in the provided DataFrame, optionally applying differential privacy.


``compute_summed_z``
~~~~~~~~~~~~~~~~
This function computes the sum of the specified explanatory variables for the outcome events.

``perform_iteration``
~~~~~~~~~~~~~~~~
This function performs an iteration of the algorithm, computing the necessary aggregates.
