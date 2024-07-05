How to use
==========

Input arguments
---------------
The input arguments for the central function consist of:

- time_col (string): The name of the column in the dataset that contains the time data.
- outcome_col (string): The name of the column in the dataset that contains the outcome data.
- expl_vars (list): A list of explanatory variables (i.e. predictors) to be used in the computation.
- organization_ids (list): A list of organization IDs that participate in the collaboration and you wish to run the algorithm on.

Python client example
---------------------

To understand the information below, you should be familiar with the vantage6
framework.
If you are not, please read the `documentation <https://docs.vantage6.ai>`_
first, especially the part about the
`Python client <https://docs.vantage6.ai/en/main/user/pyclient.html>`_.

.. code-block:: python

  from vantage6.client import Client

  server = 'http://localhost'
  port = 5000
  api_path = '/api'
  private_key = None
  username = 'org_1-admin'
  password = 'password'

  # Create connection with the vantage6 server
  client = Client(server, port, api_path)
  client.setup_encryption(private_key)
  client.authenticate(username, password)

  # When set to None it will run the algorithm on all organizations in the specified collaboration
  organization_ids = None

  input_ = {
    'method': 'central',
    'master': True,
    'kwargs': {
        'time_col': 'overall_survival_in_days',
        'outcome_col': 'event_overall_survival',
        'expl_vars': ['clin_n_1', 'index_tumour_location_oropharynx'],
        'organization_ids': organization_ids,
    },
    'output_format': 'json'
  }

  my_task = client.task.create(
      collaboration=1,
      organizations=[1],
      name='Cox proportional hazards',
      description='Cox proportional hazards model',
      image='ghcr.io/maastrichtu-cds/v6-coxph:latest',
      input=input_,
      data_format='json'
  )

  task_id = my_task.get('id')
  results = client.wait_for_results(task_id)