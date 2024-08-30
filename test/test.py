"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python test.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""

import json
from pathlib import Path
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

# get path of current directory
current_path = Path(__file__).parent

# Mock client
client = MockAlgorithmClient(
    datasets=[
        # Data for first organization
        [{
            "database": current_path / "HEAD-NECK-RADIOMICS-HN1.csv",
            "db_type": "csv",
            "input_data": {}
        }],
        # Data for second organization
        [{
            "database": current_path / "HEAD-NECK-RADIOMICS-HN1.csv",
            "db_type": "csv",
            "input_data": {}
        }]
    ],
    module="coxph"
)

# list mock organizations
organizations = client.organization.list()
print(organizations)
org_ids = [organization["id"] for organization in organizations]

# Run the central method on 1 node and get the results
central_task = client.task.create(
    input_={
        "method": "central",
        "kwargs": {
            "time_col": "overall_survival_in_days",
            "outcome_col": "event_overall_survival",
            "expl_vars": ["clin_n_1", "index_tumour_location_oropharynx"],
            "baseline_hf": True,  # Set to True to include cumulative baseline hazard function in the results
            "binning": False,  # Set to True to enable binning of event times for added privacy
            "bin_type": "Quantile",  # Set the type of binning to use for event times ("Fixed" or "Quantile")
            "differential_privacy": True,  # Set to True to enable differential privacy
            "privacy_target": "predictors",  # Set the target of the differential privacy ("predictors" or "aggregates")
            "sensitivity": 1,  # Set the sensitivity of the Cox model coefficients for differential privacy
            "epsilon": 0.3,  # Set desired epsilon value for differential privacy
            "organization_ids": org_ids

        }
    },
    organizations=[org_ids[0]],
)
results = client.wait_for_results(central_task.get("id"))
print(results)

