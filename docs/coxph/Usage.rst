How to use
==========

Input arguments
----------------
The input arguments for the central function consist of:

- ``time_col`` (string): the name of the column containing the time data.
- ``outcome_col`` (string): the name of the column containing the outcome data
  (1 = event, 0 = censored).
- ``expl_vars`` (list): the explanatory variables (predictors) of the model.
  Each entry is either a plain column name (a numeric covariate) or a
  single-entry mapping of a column name to a spec:

  - categorical: ``{"sex": {"reference": "M", "categories": {"M": ["M"],
    "F": ["F"]}}}`` gives one 0/1 model term per non-reference category,
    named ``sex[F]``;
  - scored (ordinal): ``{"stage": {"scores": {"I": 1, "II": 2, "III": 3,
    "IV": 4}, "missing": ["unknown"]}}`` gives one numeric model term
    ``stage``.

  Codes are matched as numbers or text (``1``, ``"1"`` and ``1.0`` are the
  same code); codes listed under ``missing``, and codes the spec does not
  list, are treated as missing. The model terms — not the variable names —
  index the result's ``model`` table and the order of ``covariance``.
- ``organization_ids`` (list): a list of organisation IDs that participate in
  the collaboration and you wish to run the algorithm on. When ``None`` all
  organisations are used.
- ``covariate_profiles`` (list of objects, optional): covariate profiles for
  survival curves. Each entry maps explanatory variable names to values: a
  number for a plain or scored variable, a category label for a categorical
  variable. Variables left out are taken at the pooled event-case mean. An
  entry naming an unknown variable or an unknown label raises
  ``UserInputError``.

Input rules
------------

Every node validates its data before sharing anything:

- ``time_col`` must exist and be numeric, finite and non-negative.
- ``outcome_col`` must exist and be binary: only 0 and 1 (booleans are
  allowed).
- Every ``expl_var`` source column must exist; categorical and scored
  entries are encoded into numeric model-term columns on the node before
  validation, and every model term must be numeric. Rows with missing
  values in any analysed column are dropped before the sample-size
  threshold is checked.

A column that violates these rules raises ``UserInputError`` and stops the
whole analysis; there is no silent fallback. Projects whose data uses other
codings (for example an outcome coded 1/2, or string labels) must recode
**before** calling the algorithm — recoding belongs in the project layer,
not in the vanilla algorithm.

Categorical and scored covariates
---------------------------------

With the test data (``tests/data/coxph_test_data_1.csv``, where ``sex`` is
coded ``M``/``F``), a categorical covariate and a scored one look like:

.. code-block:: python

  expl_vars = [
      "age",  # numeric covariate, as before
      {"sex": {  # categorical: one 0/1 column per non-reference category
          "reference": "M",
          "categories": {"M": ["M"], "F": ["F"]},
      }},
      {"stage": {  # ordinal: one numeric column
          "scores": {"I": 1, "II": 2, "III": 3, "IV": 4},
          "missing": ["unknown"],  # optional
      }},
  ]

The model terms are ``["age", "sex[F]", "stage"]``: they are the names in
the result's ``model``, the order of ``covariance``, ``beta``, ``centre``
and ``scale``, and the keys of the partials' ``sum``/``sum_squares``. A
covariate profile names a categorical variable by one of its labels
(``{"sex": "F"}``); central expands the label to the 1/0 values of that
variable's terms before computing the curves.

Output fields
-------------

The central function returns a dictionary with:

- ``model``: a JSON object indexed by model term (a variable name, or
  ``<name>[<label>]`` for a categorical variable) with columns ``Coef``,
  ``Exp(coef)``, ``SE``, ``lower_CI``, ``upper_CI``, ``Z`` and ``p-value``.
  The coefficients are on the original covariates: nodes fit on the
  standardised ``(x - centre) / scale`` and the central function transforms
  the reported statistics back.
- ``overall_p_value``: the overall Wald test p-value.
- ``aic``: the Akaike Information Criterion.
- ``degrees_of_freedom``: the number of model parameters.
- ``warnings``: a list of human-readable warning strings.
- ``converged`` (bool): whether the Newton-Raphson optimiser converged.
- ``n_iterations`` (int): the number of iterations performed.
- ``convergence_tolerance`` (float): the tolerance that declared
  convergence — the primary step tolerance ``1e-6``, or the secondary
  stability tolerance ``1e-3`` when the log-likelihood gain stays at
  rounding level for two consecutive accepted evaluations (coarse
  aggregate rounding can stall the step above the primary tolerance
  without the fit having moved).
- ``log_likelihood`` (float): the Breslow partial log-likelihood at the
  reported coefficients.
- ``log_likelihood_null`` (float): the log-likelihood at ``beta = 0``.
- ``lr_statistic`` / ``lr_p_value`` (float): the likelihood-ratio test
  statistic ``2 * (ll - ll_null)`` and its chi-squared p-value with
  ``degrees_of_freedom`` degrees of freedom.
- ``n_events`` (int): the pooled number of event cases.
- ``covariance``: the covariance matrix of the coefficients (original
  covariate scale).
- ``algorithm_version`` (str): the algorithm package version the result was
  computed with.
- ``curve_time_sig_figs`` (int or null): the number of significant figures
  the ``baseline_cumulative_hazard`` and ``survival_curves`` time keys are
  rounded to (``None`` when the exact event times are reported).

``overall_p_value`` is the Wald test; ``lr_p_value`` is the likelihood-ratio
test. Convergence is declared when the largest Newton step falls to ``1e-6``,
or when the log-likelihood gain stays at rounding level for two consecutive
accepted evaluations while the step stays below ``1e-3``; the tolerance that
fired is reported as ``convergence_tolerance``. When the log-likelihood
decreases after a Newton step the optimiser halves the step and retries;
the reported statistics always belong to the last accepted evaluation (the
iteration budget is 20 round-trips).

Guard-effect summary
--------------------

The result carries a ``privacy_guards`` object — an aggregate summary of the
guards that were active on the nodes (configuration, not data; there is no
per-node breakdown): ``active`` (bool), ``max_min_risk_set_change`` (the
largest ``k`` any node used), ``time_binning`` (bool),
``max_min_time_event_count`` (the largest minimum per-time event count any
node used), ``max_min_risk_set_size`` (the largest configured risk-set
size floor any node used; ``None`` when no node configured one — the floor
only takes effect when ``k > 1``) and ``max_held_event_fraction`` (the
largest share of a node's events scored against a held, unchanged risk
set — see ``Privacy.rst``). When any guard was active a warning is
appended: the estimates are then approximate (see the bias study in
``Validation.rst``). A further warning is appended when the held-event
fraction exceeds 0.25, and central aborts with ``UserInputError`` above
``COXPH_MAX_HELD_EVENT_FRACTION`` (default 0.5, read centrally; an empty
value disables the abort).

Survival curves
----------------

The result also contains:

- ``baseline_cumulative_hazard``: the Breslow baseline cumulative hazard
  ``H0(t) = sum_{t_j <= t} d_j / S0(t_j)`` over the event-time grid (the
  binned grid when time binning is on), where ``d_j`` is the pooled event
  count per grid time and ``S0(t_j)`` the risk-set sum at the reported
  coefficients. The baseline is at the covariate centre: the pooled
  event-case mean covariate profile.
- ``survival_curves``: survival functions ``S(t | x) = S0(t) ** exp(beta .
  (x - centre) / scale)``. Key ``"0"`` is the baseline (the covariate
  centre); every requested ``covariate_profiles`` entry follows under its
  1-based index.

Both are computed centrally from quantities that are already aggregated —
no new data leaves a node for them. The time keys are rounded to
``COXPH_CURVE_TIME_SIG_FIGS`` significant figures (default 6, read
centrally; an empty value reports the exact event times); times that merge
into one key keep the value at the largest original time.

The Wald statistic is reported as ``Z = Coef / SE`` and the p-value as
``p = 2 * Phi(-|Z|)``. When ``converged`` is ``False`` a warning is appended
advising that the SE/p-values may be unreliable; statistics are reported
at the last evaluated beta (the Newton step is not applied when the
iteration budget is exhausted, so Coef, SE and AIC stay consistent).

All statistics in the model table are full-precision floats (no
rounding); presentation rounding is a client concern.

Node configuration (algorithm_env)
----------------------------------

Privacy guards are configured through node environment variables
(``algorithm_env``):

=================================== =========== ============ ========================
Variable                            Type        Default      Description
=================================== =========== ============ ========================
``SAMPLE_SIZE_THRESHOLD``           int         10           Minimum rows and events
                                                             (strictly greater).
``COXPH_TIME_BIN_WIDTH``            float       (disabled)   Event-time bin width. When
                                                             set, times are coarsened
                                                             to ``floor(t/w)*w``.
``COXPH_MIN_RISK_SET_CHANGE``       int         5            Minimum risk-set change
                                                             (``k``). Consecutive
                                                             shared aggregates differ
                                                             by 0 or ``>= k``. Set to
                                                             ``1`` to disable tail
                                                             censoring and the
                                                             risk-set guards (jump
                                                             guard and size floor).
``COXPH_PRIVACY_LEVEL``             string      ``standard`` The hardening level:
                                                             ``minimal``, ``standard``
                                                             or ``strict``. See
                                                             ``Privacy.rst``.
``COXPH_MAX_COVARIATES``            int         20           Maximum number of
                                                             explanatory variables.
``COXPH_MAX_GRID_POINTS``           int         1000         Maximum pooled event-time
                                                             grid points.
``COXPH_MIN_RISK_SET_SIZE``         int         (preset)     Absolute floor on the size
                                                             of a released risk set: while
                                                             fewer individuals are at risk,
                                                             the node holds the last
                                                             released aggregates. Only
                                                             effective when ``k > 1``.
``COXPH_CURVE_TIME_SIG_FIGS``       int         6            Significant figures of the
                                                             reported curve time keys (read
                                                             centrally). An empty value
                                                             reports the exact event times.
``COXPH_MAX_HELD_EVENT_FRACTION``   float       0.5          Central abort cap on the
                                                             held-event fraction (read
                                                             centrally). An empty value
                                                             disables the abort.
=================================== =========== ============ ========================

See ``docs/coxph/Privacy.rst`` for the privacy implications of these
settings.

Python client example
----------------------

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
