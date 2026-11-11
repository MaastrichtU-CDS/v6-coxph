# Changelog

## 2.2.0

Additive wire change: ``expl_vars`` entries may now be categorical or scored
covariates, not only plain column names. A list of names stays valid, so
existing requests keep working.

- ``expl_vars`` accepts, next to plain column names, single-entry mappings of
  a column name to a spec: ``{"sex": {"reference": "M", "categories": {"M":
  ["M"], "F": ["F"]}}}`` gives one 0/1 model term ``sex[F]`` per
  non-reference category; ``{"stage": {"scores": {"I": 1, "II": 2},
  "missing": ["unknown"]}}`` gives one numeric model term ``stage``. The
  model terms — the names in the result's ``model``, the order of
  ``covariance``/``beta``/``centre``/``scale`` and the keys of the partials'
  ``sum``/``sum_squares`` — replace the variable names downstream, and
  ``COXPH_MAX_COVARIATES`` counts model terms, not entries.
- The node encodes the categorical and scored covariates into numeric term
  columns before validation (``encode_covariates``); the source columns are
  kept. Codes compare through ``code_key``: a number and the text that reads
  as one are the same code.
- Every category of a categorical covariate — the reference included — must
  have more rows and more events than ``SAMPLE_SIZE_THRESHOLD`` on each node,
  otherwise ``PrivacyThresholdViolation`` (``check_category_sizes``). Scored
  variables are numeric covariates and get no category check.
- A code the spec does not list is treated as missing and fails the run with
  ``UserInputError`` only after every threshold passed.
- ``covariate_profiles`` values may be numbers (plain and scored variables)
  or category labels (categorical variables); central expands a label to the
  1/0 values of that variable's terms.
- ``algorithm_store.json``: the ``expl_vars`` argument type is ``json`` in
  all three functions.

## 2.1.0

Non-breaking internal change; the wire contract (``algorithm_store.json``
function signatures and result keys) is unchanged.

- All payloads that cross the wire are validated with Pydantic models in
  ``v6-cox-ph/miscellaneous.py``: the user input (``CoxPHInput``), the
  payload central sends to ``perform_iteration`` (``IterationInput``, new
  in 2.1.0) and the partial results (``SummedZResult`` and
  ``IterationResult``, new in 2.1.0). The partials build their result
  dicts from the result models, so the keys and nesting of what travels
  over the wire are pinned in one place; central validates the received
  results with the same models and attributes errors by the
  ``organization_id`` the result carries.
- ``agg3`` is now also checked for finiteness (2.0.0 checked its shape
  only).
- The node's privacy settings (``SAMPLE_SIZE_THRESHOLD``,
  ``COXPH_TIME_BIN_WIDTH``, ``COXPH_MIN_RISK_SET_CHANGE``) are validated by
  a frozen model with declarative bounds; invalid settings raise
  ``UserInputError`` as before.
- The risk-set aggregates that travel to central are quantised to float32
  precision (about 7 significant figures) at serialisation — always on,
  applied after the configurable aggregate rounding — which caps the size of
  the JSON payloads. The log-likelihood decrease test that triggers
  step-halving now uses a tolerance derived from the effective released
  precision (the coarser of the configured rounding and the wire
  quantisation), so rounding jitter cannot masquerade as a real decrease.
- A new risk-set size floor, ``COXPH_MIN_RISK_SET_SIZE`` (preset-carried: off
  under ``minimal``, 50 under ``standard``, 100 under ``strict``; an empty
  string disables it): while fewer than this many individuals are at risk,
  the partial holds the last released aggregates instead of releasing a
  smaller set. Like the minimum-change guard it only takes effect when
  ``COXPH_MIN_RISK_SET_CHANGE`` > 1, so ``k = 1`` disables both. The
  ``privacy_guards`` result summary gains ``max_min_risk_set_size`` and the
  warning names the configured floor when it is active.
- The time keys of ``baseline_cumulative_hazard`` and ``survival_curves`` are
  rounded to ``COXPH_CURVE_TIME_SIG_FIGS`` significant figures (default 6,
  read centrally; an empty string reports the exact event times); times that
  merge into one key keep the value at their largest original time. The
  result gains a ``curve_time_sig_figs`` field reporting the setting in use.
- The sample-size check also enforces an admission bound while the risk-set
  size floor is in effect: a node must have at least
  ``max(reconstruction_row_factor * k, 3 * min_risk_set_size)`` rows — 150
  rows at ``standard``, 300 at ``strict``; ``minimal`` is unchanged. Nodes
  below the bound are refused the same fail-closed way as below-threshold
  nodes.
- The result's ``privacy_guards`` summary gains ``max_held_event_fraction``:
  the maximum across nodes of the share of a node's events scored against a
  held (unchanged) risk set. A warning is appended above 0.25.
- Central aborts a run whose held-event fraction exceeds
  ``COXPH_MAX_HELD_EVENT_FRACTION`` (float in (0, 1], default 0.5, read
  centrally; an empty string disables the abort) with a ``UserInputError``
  naming the cap and the remedies.
- Convergence is declared when the largest Newton step falls to ``1e-6``,
  or when the log-likelihood gain stays at rounding level for two
  consecutive accepted evaluations while the step stays below ``1e-3``
  (the secondary tolerance lets the ``strict`` preset's coarse aggregates
  converge). The result gains ``convergence_tolerance``, the tolerance that
  fired.
- The small-count time suppression is now consistent with the risk sets:
  ``perform_iteration`` moves the event rows' times up to their merged time
  (censored rows keep their time), so the analysed rows, the shared
  event-time grid and the risk sets describe the same event times. The
  ``times`` payload of ``compute_summed_z`` is unchanged.

## 2.0.0 (2026-10-05)

BREAKING wire-contract changes (a major version bump; the commit messages
of the changes below start with ``BREAKING(wire):``):

- ``perform_iteration`` gains the ``outcome_col``, ``centre`` and ``scale``
  arguments. Nodes drop rows with a missing outcome and fit on the
  standardised covariates ``(x - centre) / scale``; the pooled covariate
  centre and spread over the event cases are shared with the nodes,
  rounded to two significant figures. The partial likelihood is invariant
  to this shared affine transform; centring prevents overflow from large
  offsets and scaling fixes the Hessian conditioning from large spreads.
  Central back-transforms the reported coefficients, standard errors and
  covariance.
- ``compute_summed_z`` additionally returns the sum of squares of the
  covariates over the event cases (needed for the pooled scale) and the
  node's privacy settings (configuration, not data).
- ``get_unique_event_times`` is removed. A node below the sample-size
  threshold fails the whole analysis with ``PrivacyThresholdViolation``
  instead of being excluded: there is no eligibility round and no retry
  loop, and a missing column raises ``UserInputError``. Researchers select
  organisations explicitly via ``organization_ids``.
- Every partial result carries the ``organization_id`` it was computed
  for, and ``central`` attributes errors by it.
- The ``model`` result field is a JSON object (previously a JSON string),
  and ``included_organizations`` / ``excluded_organizations`` are removed
  from the result.

Non-breaking additions in the same release:

- Survival curves: ``central`` optionally takes ``covariate_profiles``
  and returns ``baseline_cumulative_hazard`` (Breslow, at the pooled
  event-case covariate mean) and ``survival_curves`` per profile — all
  computed centrally from existing aggregates; no new partial.
- The optimiser tracks the partial log-likelihood centrally, halves the
  Newton step when it decreases (keeping the last accepted state, so the
  reported statistics always belong to one evaluated beta) and uses a
  20-round-trip budget. New result fields: ``log_likelihood``,
  ``log_likelihood_null``, ``lr_statistic``, ``lr_p_value``, ``n_events``,
  ``covariance``, ``algorithm_version``, ``baseline_cumulative_hazard``,
  ``survival_curves`` and the aggregate ``privacy_guards`` summary
  (``active``, ``max_min_risk_set_change``, ``time_binning``) with a
  warning when the guards were active.
- The risk-set aggregates are computed vectorised
  (``searchsorted``/``bincount``/reverse cumsum, the jump guard derived
  from the bucket counts) instead of with T x N boolean masks; benchmark
  at 100k rows x 10k event times x 10 covariates: 89.3 s / 717 MB before,
  0.12 s / 28 MB after. ``agg2`` travels as a plain (T x p) list.
- Input validation is enforced on the node: the time column must be
  numeric, finite and non-negative; the outcome column must be binary;
  violations raise ``UserInputError`` and stop the analysis. Every node
  also fails closed unless every dispatched organisation answered.

## 1.0.0

Initial release of the hardened federated Cox proportional hazards model
(Breslow ties) for vantage6 4.14–4.15.
