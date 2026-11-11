Implementation
==============

Overview
--------
The algorithm fits a Cox proportional hazards model with Breslow ties by
federated Newton–Raphson, following the approach of WebDISCO (Lu et al.,
2015; see ``References.rst``). The partial likelihood, its gradient and its
Hessian depend on the data only through sums over the event cases and over
the risk set at each event time. Such sums add up across data stations: each
data station computes them on its own rows, and the central aggregator adds
them up and fits the model. No row-level data leaves a data station.

One run proceeds as follows::

    researcher -> central
      1. compute_summed_z      once, on every selected organisation
           <- per covariate: sum and sum of squares over the event cases
           <- per-time event counts, privacy settings
      2. central builds the pooled event-time grid, centre and scale
      3. perform_iteration     once per Newton–Raphson round
           -> beta, centre, scale, event-time grid
           <- risk-set sums S0, S1, S2 per grid time
      4. central takes a Newton step and tests convergence; back to 3
      5. central computes the model results, tests and survival curves
    central -> researcher

How a run proceeds
------------------

Input and organisations
~~~~~~~~~~~~~~~~~~~~~~~
``central`` validates the user input (see `Wire validation`_), rejects more
explanatory variables than ``COXPH_MAX_COVARIATES`` allows and selects the
organisations: the ``organization_ids`` given, or every organisation in the
collaboration. An unknown organisation ID raises ``UserInputError``.

Round 1: summed statistics and event times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``central`` dispatches ``compute_summed_z`` to every selected organisation.
Each data station first prepares its data through the shared guard chain
(parent-task check, covariate encoding and input validation, dropping
incomplete rows, sample-size threshold, category threshold) and bins and
tail-censors the time column; see ``Privacy.rst``. Over its event cases it
then returns the sum and the sum of squares of each explanatory variable,
together with the per-time event counts (small counts are merged into the
following time) and its privacy settings.

``central`` requires a result from every organisation it dispatched to: a
data station that fails, for example with ``PrivacyThresholdViolation``,
fails the whole run rather than leaving a model fitted on a subset. From the
results ``central`` builds:

- ``z_sum``: the pooled sum of each explanatory variable over the event
  cases;
- the pooled event-time grid ``t_1 < ... < t_T`` with the pooled event count
  ``d_j`` at each grid time (``aggregated_time_events``); a grid larger than
  ``COXPH_MAX_GRID_POINTS`` raises ``UserInputError``;
- the covariate centre and scale (see `Standardisation`_);
- the ``privacy_guards`` summary of the guards that were active on the data
  stations.

Categorical and scored covariates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``expl_vars`` entries may be plain column names or single-entry mappings to
a categorical or scored spec. Before validation, each node encodes them
(``encode_covariates`` in ``privacy_guards.py``): a categorical variable
maps each value to its category label and writes one float column
``<name>[<label>]`` per non-reference category (1.0/0.0); a scored
variable maps each value to its score in the column named after the
variable. Codes compare through ``code_key``: a number and the text that
reads as one are the same code. Listed ``missing`` codes and unlisted
codes become NaN in all of the variable's columns, so those rows are
dropped with the other incomplete rows, identically in every partial. The
source columns are kept. The model terms — one per plain or scored variable, one
per non-reference category — replace the variable names in everything
downstream: the partials' ``sum``/``sum_squares`` keys, the standardisation
vectors, the result's ``model`` index and the order of ``covariance``.

Newton–Raphson rounds
~~~~~~~~~~~~~~~~~~~~~
Starting at ``beta = 0``, every round ``central`` dispatches
``perform_iteration`` with the current ``beta``, the centre, the scale and
the event-time grid. Each data station prepares its data as in round 1, so
it analyses the same rows, validates the payload, moves its event times to
their merged times and standardises its covariates as
``x* = (x - centre) / scale``. For every grid time ``t_j`` it then computes
the risk-set sums over the rows still at risk (``time >= t_j``)::

    S0(t_j) = sum exp(beta . x*)
    S1(t_j) = sum x* exp(beta . x*)
    S2(t_j) = sum x* x*^T exp(beta . x*)

The risk-set guards may hold the previous risk set instead of releasing a
small change, and the sums are rounded to the configured number of
significant figures and quantised to float32 precision before they leave
the data station. When the guards are active the estimates are therefore
approximate (see ``Privacy.rst`` and the bias study in ``Validation.rst``).

``central`` validates each result, checks that the released risk-set sizes
are plausible (finite, non-negative, non-increasing along the grid and
positive at the first grid time) and adds the sums up across data stations.
The sum of the data stations' risk-set sums is the risk-set sum of the
pooled data, so the fit equals the fit on the pooled data up to the guards
and the rounding. In the first round, at ``beta = 0``, ``central`` also
computes the held-event fraction and aborts the run when it exceeds
``COXPH_MAX_HELD_EVENT_FRACTION``.

Each round ends with the Newton step and the convergence test described in
`Convergence`_.

Model mathematics
-----------------
The mathematics of the model lives in ``v6-cox-ph/coxph_logic.py``,
separate from the orchestration in ``central.py`` and ``partial.py``. These
functions are not vantage6 functions a user or node can call: ``central``
imports them and runs them on the central aggregator, on aggregates it has
already collected. Besides the derivatives, the module holds the partial
log-likelihood, the covariate standardisation and its back-transform, the
model results and the survival curves.

Partial likelihood and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With Breslow ties, all ``d_j`` events at grid time ``t_j`` share the same
risk set (``lifelines`` uses Efron ties instead; see ``Validation.rst``).
The partial log-likelihood, its gradient and its Hessian are::

    l(beta) = beta . z_sum - sum_j d_j log S0(t_j)
    g(beta) = z_sum - sum_j d_j S1(t_j) / S0(t_j)
    H(beta) = - sum_j d_j [ S2(t_j) / S0(t_j) - S1(t_j) S1(t_j)^T / S0(t_j)^2 ]

and the Newton–Raphson update is ``beta <- beta - H^-1 g``. All three need
only ``z_sum`` and ``d_j`` from round 1 and the summed ``S0``, ``S1`` and
``S2`` of the current round, which is why the data stations never have to
share rows.

Standardisation
~~~~~~~~~~~~~~~
Before iterating, ``central`` derives the pooled covariate centre and scale
over the event cases from the round 1 sums, with ``n_events`` the pooled
number of event cases::

    centre = z_sum / n_events
    scale  = sqrt(sum_squares / n_events - centre^2)

Both are rounded to two significant figures, so the data stations learn
less about the pooled covariate distribution; a covariate with no spread
over the event cases gets scale 1. The data stations fit on
``x* = (x - centre) / scale`` and ``central`` uses the matching
``z_sum* = (z_sum - n_events * centre) / scale``. The partial likelihood is
invariant to this affine transform: the centre cancels between the two
terms of ``l(beta)`` and the scale only reparametrises ``beta``. Centring
prevents ``exp`` overflow from large covariate offsets, and scaling keeps
the Hessian well conditioned when the covariates have very different
spreads.

The reported statistics are transformed back to the original covariates:
``Coef = beta* / scale``, ``SE = SE* / scale`` and
``covariance = C* / (scale scale^T)``; the Z statistics and p-values are
invariant.

``compute_derivatives``
~~~~~~~~~~~~~~~~~~~~~~~
This function computes the primary and secondary derivatives (the gradient
``g`` and Hessian ``H`` above) needed to compute the maximum likelihood
estimates of the model parameters. ``central`` calls it for every accepted
evaluation (see `Convergence`_) with the summed risk-set aggregates, the
per-time event counts and ``z_sum*``.

Convergence
-----------
``central`` runs at most ``MAX_ITERATIONS`` (20) rounds, each one round trip
to the data stations. Every round it evaluates the partial log-likelihood at
the current ``beta`` from the aggregates it has just collected, so this
costs no extra round trip; the value at ``beta = 0`` is the null
log-likelihood.

- **Step halving.** A decrease of the log-likelihood only becomes visible in
  the round after a step. When the log-likelihood has decreased by more than
  the rounding tolerance (below) compared with the last accepted
  evaluation, the evaluation is rejected, the step is halved and the next
  round evaluates the halved step from the last accepted ``beta``.
- **Accepted evaluation.** Otherwise ``central`` computes the derivatives,
  keeps ``beta``, the Hessian and the summed ``S0`` as the accepted state
  and solves for the Newton step.
- **Primary tolerance.** Convergence is declared when the largest absolute
  step is at most ``1e-6``. The step is then not applied, so the reported
  ``beta``, Hessian and ``S0`` all belong to the same evaluation.
- **Secondary tolerance.** The data stations release rounded aggregates, so
  the centrally computed log-likelihood carries a rounding jitter that can
  stall the step above ``1e-6`` without the fit moving. Convergence is also
  declared when the log-likelihood has changed by no more than the rounding
  tolerance over two consecutive accepted evaluations while the largest
  step is at most ``1e-3``. The rounding tolerance is the sum over the data
  stations of ``events * 10^(1 - N)``, where ``N`` is the coarser of the
  data station's configured number of significant figures and the float32
  wire precision (seven significant figures).
- **Stopping without convergence.** A singular Hessian, a non-finite step or
  an exhausted iteration budget ends the loop with ``converged = False`` and
  a warning; the results are reported at the last accepted evaluation.

The tolerance that fired is reported as ``convergence_tolerance``.

Results computed centrally
--------------------------
When the loop ends, ``central`` computes the results from the last accepted
evaluation, first in the standardised space and then transformed back (see
`Standardisation`_). No further partial runs and no new data leaves a data
station.

- **Coefficients.** The covariance matrix is ``(-H)^-1`` and the standard
  errors are the square roots of its diagonal. Each coefficient gets the
  Wald statistic ``Z = Coef / SE``, the p-value ``2 * Phi(-|Z|)`` and the
  95% confidence interval ``exp(Coef +/- 1.96 * SE)`` of its hazard ratio.
- **Overall tests.** The overall Wald test uses ``beta^T (-H) beta`` and the
  likelihood-ratio test ``2 * (l - l_null)``, both against a chi-squared
  distribution with one degree of freedom per covariate. The AIC is
  ``-2 l + 2 p`` for ``p`` covariates.
- **Survival.** The Breslow baseline cumulative hazard is
  ``H0(t) = sum_{t_j <= t} d_j / S0(t_j)`` at the reported ``beta``, with the
  covariate centre as the baseline profile; the survival curve of a
  covariate profile ``x`` is ``S(t | x) = exp(-H0(t) * exp(beta . x*))``.
- **Warnings.** A coefficient or standard error above 10 in absolute value
  (in the standardised space) or not finite points to perfect prediction
  and adds a warning, as do active privacy guards, a large held-event
  fraction and non-convergence.

``Usage.rst`` lists the output fields.

Wire validation
---------------
Every payload that crosses a trust boundary as JSON with a schema known in
advance is validated against a Pydantic model defined in
``v6-cox-ph/miscellaneous.py``: the user input (``CoxPHInput``), the
payload central sends to ``perform_iteration`` (``IterationInput``) and the
partial results (``SummedZResult``, ``IterationResult``). The partials
build their result dicts from the result models, so the keys and nesting of
what travels over the wire are pinned in one place; central validates the
received results with the same models, attributing any error by the
``organization_id`` the result carries. ``algorithm_store.json`` remains
the user-facing wire contract: function names, arguments and types.

Central
--------
The central part is responsible for the orchestration and aggregation of the algorithm.

``central``
~~~~~~~~~~~~~~~~
The central part is responsible for the following tasks:

- Validate the input and select the organisations.
- Request the summed z-statistics for the explanatory variables (i.e. the
  predictors) and, with them, the per-time event counts in each data
  station.
- Build the pooled event-time grid and the covariate standardisation.
- Orchestrate the Newton–Raphson rounds in the data stations and add up the
  risk-set sums they return.
- Pass the summed aggregates to ``compute_derivatives`` (see
  `Model mathematics`_) and take the Newton step.
- Compute the model results, the overall tests and the survival curves,
  transformed back to the original covariates.

Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``compute_summed_z``
~~~~~~~~~~~~~~~~~~~~
This function runs once per analysis. Over the node's event cases it returns
the sum and the sum of squares of each explanatory variable, together with
the per-time event counts (after small-count suppression) and the node's
privacy settings. The covariates are not standardised here: the centre and
scale are derived from these sums.

``perform_iteration``
~~~~~~~~~~~~~~~~~~~~~
This function runs once per Newton–Raphson round. It standardises the
covariates with the centre and scale it receives and returns, for every
point of the pooled event-time grid, the risk-set sums ``S0``, ``S1`` and
``S2`` at the ``beta`` it receives, after the risk-set guards, rounding and
float32 quantisation.
