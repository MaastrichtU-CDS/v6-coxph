Privacy
=======

Guards
------

The partial functions run on each data station and share only aggregated
quantities with the central aggregator. Several guards protect against the
main leakage channels of the federated design. All guards are implemented in
``v6-cox-ph/privacy_guards.py`` and applied in the same order in every
partial: parent-task check, settings load, covariate encoding and
explanatory-variable validation (``compute_summed_z`` and
``perform_iteration``), NaN row dropping, sample-size threshold, category
threshold, iteration-input validation (``perform_iteration`` only),
time-column preparation (binning, tail censoring), and the risk-set guards
(``perform_iteration`` only). On top of these mandatory guards,
the node's privacy level adds the reconstruction row bound, small-count
suppression of the shared event counts, an absolute floor on the size of a
released risk set and rounding of the released aggregates (see
Configuration below); two central-side resource caps bound the size of the
run (see Resource caps below), and the reported curve time keys are
coarsened centrally (see Curve time-key coarsening below).

Integrity / poisoning (accepted residual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm defends **confidentiality**, not **integrity**, against a
malicious participant: vantage6 has no secure aggregation, so a
colluding or compromised node can return well-formed but false
aggregates and bias the fit, undetectably in general. As defence in
depth against a *broken or grossly tampering* node, central rejects
structurally impossible risk-set sequences (non-negative, non-increasing
along the grid, positive first release) — explicitly not a poisoning
defence; a careful adversary can stay inside the bounds. Detection of a
careful malicious node requires secure or robust aggregation, which is
out of scope for the vanilla algorithm.

Parent-task guard
~~~~~~~~~~~~~~~~~

Every partial calls ``ensure_spawned_by_central``. The container JWT carries
the running ``task_id``; the partial looks up that task and requires a
non-empty ``parent`` field. The vantage6 server only sets ``parent_id`` for
container-created sub-tasks, so a user task created directly (calling a
partial without going through ``central``) has ``parent = None`` and is
refused with a ``PrivacyViolation``. A failure to look up the task fails
closed with an ``AlgorithmError``. The guard is skipped (with an info log)
for the ``MockAlgorithmClient`` used in tests.

Sample-size threshold
~~~~~~~~~~~~~~~~~~~~~

Each node must have strictly more than ``SAMPLE_SIZE_THRESHOLD`` rows **and**
events (default 10). A node below the threshold raises
``PrivacyThresholdViolation`` and the whole analysis stops: there is no
automatic exclusion. Researchers select the organisations explicitly via
``organization_ids``. The failing node is identifiable from the run status —
the same information today's exclusion marker used to leak — which is an
accepted consequence of failing closed.

The threshold bounds *participation*, not *reconstruction*: a node whose
entire content enters a single released risk set can be reconstructed from
the aggregates even though it passes the threshold. Each node must
therefore also have at least ``factor * k`` rows, where ``k`` is
``COXPH_MIN_RISK_SET_CHANGE`` and ``factor`` comes from the privacy level
(10 at ``standard``, 20 at ``strict``; no bound at ``minimal``) or from
``COXPH_RECONSTRUCTION_ROW_FACTOR`` when that variable is set. As with the
sample-size threshold, the warning on failure names the configured bound,
never the actual row count.

While the risk-set size floor is in effect, the admission bound
additionally requires ``3 x R`` rows, where ``R`` is the floor
(``COXPH_MIN_RISK_SET_SIZE``): a node that small would have (nearly) its
whole content inside the smallest released risk set, which the floor alone
cannot hide. A node must therefore have at least ``max(factor * k, 3 * R)``
rows. At the presets that is 150 rows at ``standard`` (``max(10 x 5,
3 x 50)``) and 300 rows at ``strict`` (``max(20 x 5, 3 x 100)``);
``minimal`` has no floor and is unchanged. Nodes below the bound are
refused the same fail-closed way as below-threshold nodes.

Categorical covariates
~~~~~~~~~~~~~~~~~~~~~~

A categorical covariate is encoded on the node into one 0/1 column per
non-reference category (``sex[F]`` for a variable ``sex`` with reference
``M``). Those columns are ordinary numeric covariates from there on, but
their aggregates carry category information: the ``sum`` of a 0/1 column
over the event cases is the node's number of events in that category, and
its ``agg1`` column is the number at risk in it at each event time. A
category that is small on a node is therefore visible in the released
aggregates even though no row leaves the node.

Every category of every categorical covariate — the reference category
included — must have strictly more rows and more events than
``SAMPLE_SIZE_THRESHOLD`` on each node, the same comparison the
sample-size threshold uses (``check_category_sizes``); otherwise the node
raises ``PrivacyThresholdViolation`` and the run fails closed. An empty
category fails as well: a zero is a disclosure too. The check is needed
with many-level variables, where a rare level would otherwise slip
through the node-wide threshold; researchers group rare codes into a
shared category. Scored (ordinal) variables are numeric covariates and
get no category check.

A code the spec does not list is treated as missing: the row is dropped
with the other incomplete rows. When every threshold still passes, the
run then fails with ``UserInputError`` naming the variables — but only
after every threshold passed. Raised earlier, the error would tell a
researcher whether a below-threshold node holds a code. The error names
the variables, never the codes found.

Time binning
~~~~~~~~~~~~

When ``COXPH_TIME_BIN_WIDTH`` is set to a positive value, event times are
coarsened to a regular grid ``floor(t / w) * w`` before anything else, so
event times, ``z_sum`` and risk sets are all on the same grid. Binning is
disabled by default: it is the node administrator's choice, and it should
be enabled when exact event dates are sensitive (a coarse grid cannot be
reversed to an exact date).

Tail (administrative) censoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``k = COXPH_MIN_RISK_SET_CHANGE`` (default 5), rows with ``time > t_cut``
— where ``t_cut`` is the k-th largest time on the node — are administratively
censored (``time := t_cut`` and, where the outcome column is available,
``event := 0``). This guarantees ``|R(t)| >= k`` for every non-empty shared
risk set. Setting ``k <= 1`` disables tail censoring and the whole
risk-set guard family — the jump guard and the size floor below (used by
exactness tests).

Risk-set jump guard
~~~~~~~~~~~~~~~~~~~

In ``perform_iteration``, walking the requested time grid, if moving from
``t_i`` to ``t_{i+1}`` would remove fewer than ``k`` (but more than zero)
individuals from the node's risk set, the node keeps using ``R(t_i)``
("hold"). Consecutive shared aggregates therefore differ by 0 or by at
least ``k`` individuals.

Risk-set size floor
~~~~~~~~~~~~~~~~~~~

The jump guard bounds how fast the shared risk-set aggregates may change,
not how small a released risk set may be. ``COXPH_MIN_RISK_SET_SIZE`` (the
privacy level's ``min_risk_set_size``: off at ``minimal``, 50 at
``standard``, 100 at ``strict`` — the reconstruction bound ``factor * k``)
sets an absolute floor: while fewer than that many individuals are at risk
at a grid point, the node holds the last released aggregates instead of
releasing a smaller risk set (the same hold pattern as the jump guard; no
truncation). The tail of the grid is then reported at the last released
set, which makes the estimates approximate there. The floor belongs to
the risk-set guard family: ``k <= 1`` disables it together with the jump
guard, and the first grid point is always released whatever its size.

Held-event fraction
~~~~~~~~~~~~~~~~~~~

The jump guard and the size floor hold the last released aggregates while
the true risk set shrinks below the release conditions. A run whose fit is
dominated by held risk sets is attenuated towards zero yet still reports
``converged``. Central therefore measures, per node, the share of the
node's events scored against a held (unchanged) risk set: a grid point
beyond the first is held when the released risk set is still equal to the
first release — the true risk set strictly decreases at every event time
(the event rows leave it), so a released set still equal to the first
release at a later event time is necessarily stale. The maximum across
nodes is reported as ``privacy_guards.max_held_event_fraction``.

Above 0.25 a warning is appended: the estimates may be dominated by held
risk sets. Above the cap ``COXPH_MAX_HELD_EVENT_FRACTION`` (float in
(0, 1], default 0.5, read centrally; an empty string disables the abort)
central raises ``UserInputError`` before the fit continues: the message
names the configured cap, the measured maximum and the remedies — larger
nodes, lowering the risk-set guards by policy (``COXPH_MIN_RISK_SET_CHANGE``,
``COXPH_MIN_RISK_SET_SIZE``), or running at the ``minimal`` level for
validation.

Small-count time suppression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``compute_summed_z`` shares per-time event counts, and central builds the
event-time grid for the whole run from them. A per-time count of 1
identifies a singleton: an event at that exact time belongs to a single
individual. Per-time event counts below the level's minimum
(``COXPH_MIN_TIME_EVENT_COUNT``: 2 at ``standard``, 5 at ``strict``) are
therefore merged into the *following* event time (small-cell suppression).
The events are tied at the later time, never dropped, so the total number
of events is unchanged. The suppression applies to the ``times`` payload of
``compute_summed_z`` and therefore to the event-time grid the whole run
uses, including the keys of the returned ``baseline_cumulative_hazard``
and ``survival_curves``.

``perform_iteration`` applies the same merge to the node's rows: the event
rows' times are moved up to their merged time (censored rows keep their
time), so the risk sets match the shared grid and the partial likelihood
is consistent with the suppressed counts. The move applies whenever the
minimum per-time event count is above 1, whatever the risk-set guards; the
tail censoring runs first, so a merged time never exceeds the cut-off.
The analysed rows and the total number of events are unchanged.

With high-precision event times most per-time counts are singletons and
the grid coarsens substantially; the model stays valid but the estimates
shift. Combine the suppression with ``COXPH_TIME_BIN_WIDTH`` when the
remaining grid is still too fine.

Aggregate rounding
~~~~~~~~~~~~~~~~~~

The risk-set sums ``S0``/``S1``/``S2`` released by ``perform_iteration`` are
rounded to the configured number of significant figures
(``COXPH_AGGREGATE_SIG_FIGS``: 6 at ``standard``, 4 at ``strict``; an empty
string disables rounding). This is deterministic quantisation, not noise:
it defeats exact reconstruction of values from the aggregates, but the
values are recoverable to the rounding precision. The rounding is
consistent with the existing two-significant-figure rounding of the pooled
centre and scale. Lowering the number of significant figures trades
accuracy for privacy.

The released aggregates are additionally quantised to float32 precision
(about 7 significant figures) at serialisation — always on, applied after
the configured rounding. A bare float32 cast would serialise to a longer
float64 representation, so the payload cut comes from decimal
quantisation. Where the configured rounding is coarser (``standard`` 6,
``strict`` 4) the quantisation is a no-op; at ``minimal`` it is the only
precision limit on the wire. Central accounts for the effective released
precision when it judges whether a log-likelihood change is real.

Curve time-key coarsening
~~~~~~~~~~~~~~~~~~~~~~~~~

``baseline_cumulative_hazard`` and ``survival_curves`` are keyed by the
pooled event times. Those keys are rounded to
``COXPH_CURVE_TIME_SIG_FIGS`` significant figures (default 6, read
centrally; an empty string reports the exact event times). The event-time
grid itself and the hazard values are unaffected; when rounding merges
nearby times into one key, the value at the largest original time is
kept, consistent with the small-count time suppression.

When the coarsening is on (the default), the reported times are rounded:
nearby event times merge into a single curve step, so the exact event
times are not all visible on the curves. When it is off (an empty
``COXPH_CURVE_TIME_SIG_FIGS``), the curves are keyed by the exact event
times. Note that the ``times`` payload of ``compute_summed_z`` carries
the node's (small-count-suppressed) event times exactly either way.

What the coarsening protects depends on where the aggregator runs. When
the aggregator is inside the task-sending organisation — the default
vantage6 deployment, where ``central`` runs on the researcher's own node —
the researcher can read the sub-task results, so the exact times in the
``compute_summed_z`` payloads are visible regardless of the curve
coarsening; the coarsening then bounds only what the *reported output*
reveals. When the aggregator is a separate organisation and the partials'
inputs and outputs are encrypted end-to-end against that organisation's
key, the researcher never sees the per-node aggregates; the curves are
then the only time surface in the result, and the coarsening is what
bounds the time precision of the final output.

Resource caps
~~~~~~~~~~~~~

Two central-side caps bound the compute loop and the per-node, per-round
result payload, which grows with the grid length ``T`` and with the square
of the number of covariates ``p``:

- ``COXPH_MAX_COVARIATES`` (int, default 20): central rejects a model with
  more explanatory variables.
- ``COXPH_MAX_GRID_POINTS`` (int, default 1000): central rejects a run
  whose pooled event-time grid is longer. The error names the cap and the
  remedies — enable ``COXPH_TIME_BIN_WIDTH``, or select fewer
  organisations — but never the actual grid length.

Binning the event times and running single-organisation tasks reduce the
payload further.

Logging
~~~~~~~

The algorithm container log is returned to the server with the run and can
be read by the researcher. Partials therefore log status only: never data
values, and never counts. A failed sample-size check names the configured
threshold but not the number of rows or events the node actually has.
Validation errors are re-raised without the original Pydantic message —
the chained message used to embed the offending input — and unexpected
errors inside a partial surface as an ``AlgorithmError`` naming only the
exception type.

Configuration
~~~~~~~~~~~~~

All settings are read from node ``algorithm_env`` variables via
``get_env_var``:

=================================== =========== ============
Variable                            Type        Default
=================================== =========== ============
``SAMPLE_SIZE_THRESHOLD``           int         10
``COXPH_TIME_BIN_WIDTH``            float       (disabled)
``COXPH_MIN_RISK_SET_CHANGE``       int         5
``COXPH_PRIVACY_LEVEL``             string      ``standard``
``COXPH_RECONSTRUCTION_ROW_FACTOR`` int         (preset)
``COXPH_MIN_TIME_EVENT_COUNT``      int         (preset)
``COXPH_AGGREGATE_SIG_FIGS``        int         (preset)
``COXPH_MIN_RISK_SET_SIZE``         int         (preset)
``COXPH_MAX_COVARIATES``            int         20
``COXPH_MAX_GRID_POINTS``           int         1000
``COXPH_CURVE_TIME_SIG_FIGS``       int         6
``COXPH_MAX_HELD_EVENT_FRACTION``   float       0.5
=================================== =========== ============

``COXPH_PRIVACY_LEVEL`` is the primary variable: the one variable a node
administrator sets to choose the hardening level. The level presets the
four hardening settings:

============ ========== ================ ========= ==================
Level        Row factor Min. time events Sig. figs Min. risk-set size
============ ========== ================ ========= ==================
``minimal``  off        off              off       off
``standard`` 10         2                6         50
``strict``   20         5                4         100
============ ========== ================ ========= ==================

``minimal`` applies only the mandatory guards (sample-size threshold,
``k``, tail censoring, jump guard); it is meant for validation studies, not
for sensitive data. Four advanced overrides each beat the level preset
when set: ``COXPH_RECONSTRUCTION_ROW_FACTOR`` (int, at least 1),
``COXPH_MIN_TIME_EVENT_COUNT`` (int, at least 1),
``COXPH_AGGREGATE_SIG_FIGS`` (int, at least 1; an empty string disables
rounding) and ``COXPH_MIN_RISK_SET_SIZE`` (int, at least 1; an empty string
disables the floor). Precedence is: an explicitly set variable beats the
level preset, which beats the built-in default. A node administrator
normally sets nothing — the default ``standard`` level applies. The two
caps, ``COXPH_CURVE_TIME_SIG_FIGS`` and ``COXPH_MAX_HELD_EVENT_FRACTION``
are central-side; see Resource caps, Curve time-key coarsening and
Held-event fraction above.

``COXPH_MIN_RISK_SET_CHANGE`` must be at least 1, and
``SAMPLE_SIZE_THRESHOLD + 1`` must be greater than or equal to it;
otherwise the node raises ``UserInputError`` at start-up. This prevents
a configuration where a node passes the sample-size threshold but has
too few rows for tail censoring to guarantee risk sets of size ``k``.

The parent-task guard fails closed: a client without a decodable
container token (that is not a ``MockAlgorithmClient``) raises
``AlgorithmError`` rather than silently skipping the check.

Rows with NaN in ``time_col``, ``outcome_col`` or any model term are
dropped before the sample-size threshold is checked, so thresholds apply
to the analysed rows. Explanatory variables are validated on the node:
categorical and scored entries are encoded into numeric model-term
columns first, each source column must exist, must not overlap
``time_col``/``outcome_col``, and every model term must be numeric. The
input model ``CoxPHInput`` rejects duplicate explanatory variables, a
``time_col`` equal to ``outcome_col``, and explanatory variables naming
the time or outcome columns. Central validates ``organization_ids``
against the collaboration and rejects unknown identifiers with a
``UserInputError`` before dispatching any sub-task.

Guard-effect summary
~~~~~~~~~~~~~~~~~~~~

The result carries a ``privacy_guards`` object: an aggregate summary of the
guards that were active (configuration, not data; there is no per-node
breakdown). It reports the maximum ``k`` any node used, whether time
binning was on, the maximum minimum time event count across nodes, the
maximum configured risk-set size floor across nodes (``None`` when no node
configured one; the floor only takes effect when ``k > 1``) and the
maximum held-event fraction across nodes (see Held-event fraction above).
A warning is appended when any of these is active, because the estimates
are then approximate; a further warning is appended when the held-event
fraction exceeds 0.25.

Data sharing
------------

Each data station shares, with the central aggregator only:

- the sum of each explanatory variable over the event cases, the matching sum
  of squares (used to derive the pooled covariate spread), the node's privacy
  settings (configuration, not data), and per-time event counts computed
  from the same NaN-dropped, censored DataFrame (``compute_summed_z``). The
  event counts are used by the aggregator to build the event-time grid that
  is consistent with ``z_sum`` and the risk sets; the per-time event counts
  are small-count-suppressed before they leave the node (see above). The
  sum of squares is a new aggregate leaving the node; it enjoys the same
  sample-size threshold protection as ``z_sum``;
- per event time ``t``, the risk-set sums ``S0(t) = sum exp(beta . X)``,
  ``S1(t) = sum X * exp(beta . X)`` and ``S2(t) = sum X X^T exp(beta . X)``
  (``perform_iteration``), computed on the standardised covariates
  ``(x - centre) / scale``. The pooled centre and scale are shared with every
  node in every iteration, rounded to two significant figures: a node
  therefore learns a coarse pooled event-case covariate mean and standard
  deviation, and nothing about any other individual node. The released
  risk-set sums are rounded to the configured number of significant figures
  (see above).

Two directions of sharing exist and both must be assumed visible to the
researcher that ran ``central``:

- **Node → central.** The pooled event-time grid is sent to *every* node in
  every iteration (``unique_time_events``): every node therefore learns the
  pooled, binned event times of the collaboration. Bin times
  (``COXPH_TIME_BIN_WIDTH``) when that is too much detail.
- **Sub-task results.** The sub-tasks are created under the researcher's
  identity and their results are retrievable from the server (the vantage6
  permission checks operate at the server). Every partial must therefore
  treat its entire return value as visible to the researcher — the sample
  contents below are all aggregates over at least ``k`` individuals.
- **Survival curves.** ``baseline_cumulative_hazard`` and
  ``survival_curves`` are computed centrally from the pooled event counts
  and risk-set sums that the researcher already sees; no new partial runs
  and no new data leaves a node. What the curve reveals is, per grid time,
  the pooled event count over the risk-set sum — information the shared
  aggregates already contained. The tail censoring and the risk-set guards
  carry through: every non-empty risk set behind a curve step covers at
  least ``k`` individuals on every node, and the curve time keys are
  coarsened to ``COXPH_CURVE_TIME_SIG_FIGS`` significant figures (see
  Curve time-key coarsening above).

Aggregated model coefficients are shared with the data stations by the
central aggregator during the iteration process.

Vulnerabilities to known attacks
--------------------------------

.. Differencing / reconstruction / DLG

.. _mitigated-warning:

**Reconstruction — mitigated.**
The partials no longer return row-level data or exact per-event-time sums
that allow record reconstruction by differencing. Per-event-time risk-set
sums are shared, but the jump guard, the size floor and tail censoring
ensure every shared, non-empty risk set covers at least ``k`` individuals
and consecutive aggregates differ by 0 or by at least ``k``.
**Residual risk:** an aggregator that collects the released sums over the
smallest shared risk set across several ``beta`` values obtains an exact
equation system over that group's covariate values. The guards — a larger
``k``, the reconstruction row bound, the risk-set size floor, coarser event
times and rounded, float32-precision aggregates — raise the cost of
solving that system but cannot eliminate it. Differential privacy on the
released aggregates, or secure aggregation, is the complete fix and is
planned for a future release.

**Differencing — mitigated.**
The risk-set jump guard prevents an aggregator from differencing consecutive
shared aggregates to isolate fewer than ``k`` records. **Residual risk:**
a data-station manager who changes the dataset between tasks could still
differencing across runs; data-station managers should not be allowed to
run tasks directly.

**Deep Leakage from Gradients (DLG) — mitigated.**
The shared ``S0/S1/S2`` aggregates are sums over risk sets of at least
``k`` individuals, not per-record gradients. DLG-style reconstruction from
these aggregates would recover groups rather than individuals. The central
aggregator should nonetheless be treated as a trusted party until
differential privacy or secure aggregation is in place (see the residual
risk under Reconstruction above).

**Generative Adversarial Networks (GAN) — low risk.**
Synthetic data can statistically reproduce the data underlying the model,
but without the sensitive attributes an adversary cannot assess the
authenticity of recovered records.

**Model Inversion — low risk.**
Model predictions can be used to infer an individual's outcome, but without
the sensitive information the adversary cannot verify the inference.

**Watermark Attack — to be determined.**

Choosing thresholds per analysis type
-------------------------------------

The guards scale with the analysis type, in three tiers:

- **Descriptive statistics** may run at the base sample-size threshold:
  every output covers all rows of the node.
- **Survival modelling**, such as this algorithm, requires the
  reconstruction row bound in addition: the released risk-set sums cover
  groups of rows, not all rows.
- **Gradient-iterative modelling** — regression with iterates, clustering,
  deep learning — needs differential privacy or secure aggregation from
  the start; a threshold alone is not a mitigation there.

``COXPH_PRIVACY_LEVEL`` tunes within this algorithm's tier. Which analyses
may run at which node sizes is collaboration policy, owned by the project
privacy officer. Note that fail-closed threshold failures identify the
refusing node at any tier.

For reference
-------------

- **Reconstruction**: an adversary tries to reconstruct the original dataset
  from shared model parameters.
- **Differencing**: an adversary infers information about a specific data
  point by comparing outputs with and without it.
- **Deep Leakage from Gradients (DLG)**: an adversary infers training data
  from shared gradient updates.
- **GAN**: synthetic data statistically similar to the original data.
- **Model Inversion**: inferring input data given model output.
- **Watermark Attack**: embedding a watermark in a model to identify the
  data it was trained on.
