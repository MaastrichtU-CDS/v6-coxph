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

- Request the unique event times and their counts in each data station.
- Request the summed z-statistics for the explanatory variables (i.e. the predictors) in each data station.
- Orchestrate iterations in the data stations to retrieve intermediate model parameters.
- Pass the intermediate model parameters to the compute_derivatives function.
- Compute the aggregated model parameters.

``compute_derivatives``
~~~~~~~~~~~~~~~~
This function computes the primary and secondary derivatives needed to compute the maximum likelihood estimates of the model parameters.
The function is called by the central part and is executed on the central aggregator.

Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``get_unique_event_times``
~~~~~~~~~~~~~~~~
This function retrieves unique event times and their counts from the selected database.

``compute_summed_z``
~~~~~~~~~~~~~~~~
This function computes the sum of the specified explanatory variables for the outcome events.

``perform_iteration``
~~~~~~~~~~~~~~~~
This function performs an iteration of the algorithm, computing the necessary aggregates.
