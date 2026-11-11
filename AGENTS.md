# AGENTS.md — v6-cox-ph

A guide for coding agents (Junie, Mistral Vibe, Claude Code, Copilot, …)
working in this repository. Read this before touching node-side code.

## What this repository is

A federated Cox proportional hazards model (Breslow ties) for
[vantage6](https://vantage6.ai) 4.14–4.15 (the range pinned in
``pyproject.toml`` and tested in CI). One Docker image exposes a single
``central`` aggregator function and two ``partial`` functions that run on
data stations. The partials share only aggregated quantities with the
central aggregator; no row-level data leaves a node.

## Scope

This repository is the **vanilla** Cox-PH algorithm. Project-specific
features — data filtering, recoding, project privacy tooling, other data
sources — do not belong here; they go in project branches or forks. Vanilla
never depends on project code, and project code never flows back.

## Building on this algorithm

Projects build on this algorithm in a fork or a stacked branch. To stay
compatible with that:

- Any step that removes rows runs **before** ``check_sample_size`` and
  identically in every partial: put project row selection at the marked
  ``select_rows`` hook inside the shared ``prepare_node_data``. That way
  every aggregate central combines comes from the same row set on each
  node, and rows the project removes still count against the threshold.
- Never weaken the vanilla guards: the parent-task guard, the sample-size
  threshold and the logging rules stay as they are.
- A **breaking** wire-contract change — one that breaks an existing request
  or result — is a **major** version bump, and the commit message starts
  with ``BREAKING(wire):``. An additive change — a new optional argument,
  a widened argument type that keeps the old form valid, a new result key
  — is a minor bump.

## Repository layout

```
v6-cox-ph/                 algorithm package (the hyphenated name is intentional)
  central.py               aggregator: dispatches sub-tasks, Newton-Raphson loop
  partial.py               node-side: compute_summed_z, perform_iteration
                           (both decorated, both guarded)
  coxph_logic.py           pure math: derivatives, Newton step, model results
  privacy_guards.py        pure guards: parent-task check, thresholds, binning,
                           tail censoring, vectorised risk-set aggregates with
                           the jump guard
  miscellaneous.py         pydantic models for the wire contract (input,
                           iteration input, partial results),
                           data-quality helpers
  __init__.py              exports central and the partials
algorithm_store.json       wire contract — function names, arguments, types
Dockerfile                 container image; entrypoint is
                           ``wrap_algorithm`` from vantage6.algorithm.tools.wrap
pyproject.toml             dependencies, dev/lint extras, pytest config
tests/
  unit/                    no-Docker tests (math, guards, mock pipeline;
                           reference_risk_sets.py holds the pre-vectorisation
                           mask implementation for equivalence tests)
  integration/             Docker + v6 dev network tests
  data/                    coxph_test_data_{1,2,3}.csv, node/store configs
  conftest.py              Docker/vantage6 fixtures, markers
scripts/                   benchmark_risk_sets.py, bias_study.py (not
                           collected by pytest)
docs/coxph/                Privacy.rst, Validation.rst, Usage.rst, etc.;
                           with docs/index.rst the source of the wiki
.github/workflows/         test-suite.yml (test + lint + security + docker),
                           security-scan.yml (weekly), release.yaml,
                           publish-wiki.yml (docs/ to the wiki)
.github/dependabot.yml     pip, Docker and action updates
```

## How the algorithm executes

```
user task → central → compute_summed_z
           → perform_iteration × N (Newton–Raphson)
```

- ``central`` is decorated with ``@algorithm_client`` and is the only function
  a user should call. It validates input, collects organisation IDs,
  dispatches sub-tasks, runs the Newton–Raphson loop, and returns the model.
- The two partials are decorated with ``@data(1)`` and ``@algorithm_client``.
  They run on each node and return aggregates. Their keyword arguments
  **must match** ``algorithm_store.json`` — that file is the wire contract.
  Each partial declares exactly one database there, matching ``@data(1)``.
- A node that fails the sample-size threshold raises
  ``PrivacyThresholdViolation`` in every partial, which stops the whole
  analysis; there is no automatic exclusion and no retry round. Researchers
  select organisations explicitly via ``organization_ids``.
- ``central`` builds the event-time grid and per-time event counts
  (``aggregated_time_events``) from the ``times`` returned by
  ``compute_summed_z``. It drops rows with NaN in ``expl_vars``, so its
  counts are consistent with ``z_sum`` and the risk sets.
- The covariates are standardised before iterating: ``central`` derives the
  pooled covariate centre and scale over the event cases from ``z_sum`` and
  the sum of squares (both returned by ``compute_summed_z``, rounded to two
  significant figures), nodes fit on ``(x - centre) / scale`` and ``central``
  back-transforms the reported coefficients, standard errors and covariance.
  The partial likelihood is invariant to this affine transform; centring
  prevents ``exp`` overflow and scaling fixes the Hessian conditioning.
- ``compute_summed_z`` also returns the node's privacy settings
  (configuration, not data); the result summary based on them is built in a
  later step.
- ``expl_vars`` entries are plain column names or single-entry mappings of a
  column name to a categorical spec (``reference`` + ``categories``) or a
  scored spec (``scores``, optional ``missing``); the store type of
  ``expl_vars`` is ``json`` and a list of names stays valid. The model terms
  they expand to (one per plain or scored variable, one per non-reference
  category) replace the variable names in the result's ``model``, the order
  of ``covariance`` and the partials' ``sum``/``sum_squares`` keys.
- ``central`` optionally takes ``covariate_profiles`` (validated by the
  Pydantic input model: keys must be known explanatory variables) and
  returns ``baseline_cumulative_hazard`` and ``survival_curves`` computed
  centrally from the pooled event counts and risk-set sums — no new partial,
  no new data leaves a node.
- ``perform_iteration`` receives ``outcome_col`` and drops rows with a
  missing outcome, so every partial analyses the same rows; a node that
  passed ``compute_summed_z`` cannot fail there — its threshold check is
  defence in depth only.
- The Newton–Raphson loop tests ``max|step| <= 1e-6`` *before* applying the
  step, so the reported ``beta``, the Hessian and ``summed_agg1`` are all
  evaluated at the same ``beta`` — the last *accepted* evaluation. A
  secondary tolerance also declares convergence when the log-likelihood gain
  stays at rounding level for two consecutive accepted evaluations while the
  step stays below ``1e-3`` (coarse aggregate rounding can stall the step
  above the primary tolerance without the fit having moved). The
  log-likelihood is computed centrally each iteration from the aggregates
  already collected; when it decreases the Newton step is halved and the
  previous accepted state is kept (a decrease is only visible at the next
  round-trip). Results include ``converged``, ``n_iterations`` and
  ``convergence_tolerance`` (the tolerance that fired), plus
  ``log_likelihood``, ``log_likelihood_null``, ``lr_statistic``,
  ``lr_p_value``, ``n_events``, ``covariance`` and ``algorithm_version``.
- Results must be JSON-serialisable (they travel over the vantage6 wire).
- ``MAX_ITERATIONS`` (module-level in ``central.py``, 20) controls the epoch
  budget; tests monkeypatch it to force non-convergence.

## Environment and commands

Python 3.10 or later. The container image runs Python 3.12; the code
supports ``>= 3.10`` until 3.10 is dropped deliberately (then ``requires-python``
and Black's ``target-version`` move up together). Install in editable mode
with the test (``dev``) and lint (``lint``) extras::

    pip install -e .[dev,lint]

Unit tests (no Docker, run in a few seconds)::

    pytest -m unit
    # or
    pytest tests/unit -q

Integration tests (require Docker and the ``v6`` CLI). Missing
infrastructure makes them skip locally but fail when ``CI`` or
``REQUIRE_INTEGRATION_TESTS`` is set to ``1``/``true``/``yes``; set the latter
to force failures locally::

    pytest tests/integration
    REQUIRE_INTEGRATION_TESTS=1 pytest tests/integration

Lint and format (mirrors ``.github/workflows/test-suite.yml``; Black reads
``line-length = 120`` from ``[tool.black]`` in ``pyproject.toml``)::

    black --fast --check --diff --target-version py310 v6-cox-ph/ tests/
    flake8 v6-cox-ph/ tests/ --max-line-length=120 --extend-ignore=E203,W503
    ln -s v6-cox-ph v6_cox_ph
    mypy v6_cox_ph --ignore-missing-imports --follow-imports=silent
    rm v6_cox_ph

Build the Docker image (Alpine two-stage; the base is pinned by digest and
kept current by Dependabot)::

    docker build -t v6-cox-ph:ci-test .

The runtime dependencies are pinned in ``requirements-lock.txt`` and
installed as constraints (``-c``), not hashes. Regenerate after a dependency
change with::

    uv pip compile pyproject.toml --python-version 3.12 \
        --python-platform x86_64-unknown-linux-musl -o requirements-lock.txt

The lock targets musl/Python 3.12 (what the image runs); the CI test matrix
deliberately runs glibc and unlocked to test the declared range.

## Testing conventions

- Markers: ``unit``, ``integration``, ``slow``, ``vantage6``, ``docker``
  (``--strict-markers`` is on). Unit tests are tagged ``@pytest.mark.unit``.
- Assert numbers, not just finiteness. ``compute_derivatives``,
  ``compute_model_results``, the guards and the pipeline all have exact-value
  assertions.
- ``MockAlgorithmClient`` (from ``vantage6.algorithm.tools.mock_client``)
  runs ``central`` end-to-end in-process. The parent-task guard skips it by
  an explicit ``isinstance`` check, **not** because it has no
  ``_access_token``. Any other client without a token fails closed (see
  Privacy rules). Import the central module via
  ``import_module("v6-cox-ph.central")`` because the package name has a
  hyphen.
- ``lifelines.CoxPHFitter`` is the reference, but it uses Efron ties while
  this algorithm uses Breslow ties — expect small differences and use the
  tolerances already in the tests.
- Test data lives in ``tests/data/``. Node 3 has too few events: a run that
  includes it fails closed with ``PrivacyThresholdViolation``.
- Exactness tests disable the risk-set guards with
  ``COXPH_MIN_RISK_SET_CHANGE=1``; the default-guard test asserts the
  privacy property separately.

## Privacy rules (non-negotiable for node-side code)

- **Never return row-level data** from a partial. Only aggregates leave a
  node. The result dicts of the partials are built from the Pydantic
  result models, so their keys and nesting are pinned in one place.
- **Every partial prepares its data through the shared**
  ``prepare_node_data(client, df, time_col, outcome_col, covariates, *,
  need_outcome)`` (in ``privacy_guards.py``; ``covariates`` is the resolved
  ``ResolvedCovariates`` from ``resolve_expl_vars``), which runs the guards
  in order: ``ensure_spawned_by_central`` → ``load_privacy_settings`` →
  ``encode_covariates`` (when covariates are given) →
  ``validate_expl_vars`` (when covariates are given) →
  ``drop_incomplete_rows`` → ``validate_survival_columns`` →
  ``select_rows`` (the marked hook for project row selection) →
  ``check_sample_size`` → ``check_category_sizes`` (only when the threshold
  was met) → the unlisted-code ``UserInputError`` (only when every threshold
  passed). Do not reorder or skip any of them; the unit tests assert the
  order. ``validate_iteration_input`` (``perform_iteration`` only) and
  ``prepare_time_column`` run afterwards.
  ``perform_iteration`` builds its aggregates with the vectorised
  ``guarded_risk_set_aggregates`` (``searchsorted``/``bincount``/reverse
  cumsum; the jump guard comes from the bucket counts) — the previous mask
  implementation lives on in ``tests/unit/reference_risk_sets.py`` as the
  equivalence-test reference. ``agg2`` travels as a plain (T x p) list.
- **No new partial** without a sample-size threshold, parent-task guard and
  documentation in ``Privacy.rst``.
- **Do not log data values or counts**; use ``info``/``warn`` for status
  only. The container log is returned to the researcher with the run, so a
  row or event count, especially a below-threshold one, is a leak. Name the
  configured threshold, never the actual count.
- **Do not loosen** thresholds or tolerances to make tests pass.
- Privacy settings are read from node ``algorithm_env`` via ``get_env_var``:
  ``SAMPLE_SIZE_THRESHOLD`` (default 10), ``COXPH_TIME_BIN_WIDTH`` (disabled
  by default), ``COXPH_MIN_RISK_SET_CHANGE`` (default 5, must be >= 1, and
  ``SAMPLE_SIZE_THRESHOLD + 1 >= COXPH_MIN_RISK_SET_CHANGE``). The hardening
  knobs are selected by the ``COXPH_PRIVACY_LEVEL`` preset (``minimal`` /
  ``standard`` / ``strict``) and can be overridden individually — an
  explicitly set variable beats the preset:
  ``COXPH_RECONSTRUCTION_ROW_FACTOR``, ``COXPH_MIN_TIME_EVENT_COUNT``,
  ``COXPH_AGGREGATE_SIG_FIGS`` and ``COXPH_MIN_RISK_SET_SIZE`` (the last two
  accept an empty string to switch the knob off). Central additionally reads
  ``COXPH_MAX_COVARIATES`` (default 20), ``COXPH_MAX_GRID_POINTS`` (default
  1000), ``COXPH_CURVE_TIME_SIG_FIGS`` (default 6; an empty string
  reports the exact curve times) and ``COXPH_MAX_HELD_EVENT_FRACTION``
  (default 0.5; central aborts a run whose held-event fraction exceeds it;
  an empty string disables the abort). The full semantics live in
  ``docs/coxph/Privacy.rst``.
- The parent-task guard fails closed: if the task lookup fails or the token
  is missing (and the client is not a ``MockAlgorithmClient``), raise
  ``AlgorithmError``, never proceed.
- Rows with NaN in ``time_col``, ``outcome_col`` or any model term are
  dropped before the sample-size threshold is checked.

## Coding conventions

- numpy/pandas vectorised maths; avoid per-row Python loops in the hot path.
- Pydantic validation of everything that crosses the wire: ``CoxPHInput``,
  ``IterationInput``, ``SummedZResult``, ``IterationResult``.
- vantage6 logging: ``info``/``warn``/``error`` from
  ``vantage6.algorithm.tools.util``.
- Raise vantage6 exception types: ``UserInputError``,
  ``PrivacyThresholdViolation``, ``PrivacyViolation``, ``AlgorithmError``.
- UK English in prose, comments, docstrings and log messages (organisation,
  behaviour, optimise). Identifiers keep vantage6's US spelling and must not
  be "corrected": ``organization_ids``, ``client.organization_id`` and
  ``client.organization.list()`` are part of the wire contract or the
  vantage6 API.
- Black formatting, line length 120, target ``py310`` (configured in
  ``[tool.black]`` in ``pyproject.toml``); flake8 uses the same 120 limit.

## Branches, docs and releases

- ``standard-coxph`` is the default branch and the PR target.
- Semantic versioning. A **breaking** wire-contract change (one that breaks
  an existing request or result — partial signatures, result keys,
  ``algorithm_store.json``) is a major bump and its commit message starts
  with ``BREAKING(wire):``. An additive change — a new optional argument,
  a widened argument type that keeps the old form valid, a new result key
  — is a minor bump.
- Releases are git tags. ``release.yaml`` publishes
  ``ghcr.io/<owner>/<repository>-<branch>:<tag>`` where ``<branch>`` is the
  branch whose **tip** is the tagged commit; it fails clearly when none or
  several branches match, and a ``workflow_dispatch`` input is the escape
  hatch for a tag that is no longer a branch tip.
- Update ``docs/coxph`` and ``algorithm_store.json`` whenever a signature or
  output field changes.
- The wiki is generated from ``docs/`` (``index.rst`` becomes *Home*, every
  ``docs/coxph`` page keeps its name): edit ``docs/``, never the wiki, whose
  web edits are overwritten. Keep the docs to plain reStructuredText that
  pandoc converts (``code-block`` works; Sphinx roles such as ``:ref:`` and
  ``:doc:`` do not) and refer to another page as ``Privacy.rst``, which the
  conversion turns into a wiki link. The README stays a short overview that
  links to the wiki.
- CI lives in ``.github/workflows`` and runs with ``permissions:
  contents: read`` (only the wiki job has ``contents: write``):
  - ``test-suite.yml`` (on pushes to the default branch, every pull request
    and manual dispatch) runs the ``test`` matrix (Python 3.10/3.12 ×
    vantage6 4.14/4.15), ``lint``, ``security`` (Bandit ``-ll`` and
    ``pip-audit``; both must be able to fail — a finding is fixed or
    suppressed with a justified ``# nosec`` comment, never ``|| true``) and
    a ``docker`` build + import smoke-test job.
  - ``security-scan.yml`` runs weekly (Mondays): ``pip-audit`` on
    ``requirements-lock.txt`` (suppressions in ``.pip-audit-ignore``, each
    with a justification and an expiry date — currently the exact pins
    ``click``, ``cryptography`` and ``pyjwt`` imposed by
    ``vantage6-common``), Bandit and Trivy (HIGH/CRITICAL,
    ``ignore-unfixed``) on the latest released image; failures file a
    **private draft security advisory** (GHSA), never a public issue — the
    advisory is published only once the finding is fixed and released. The
    built-in ``GITHUB_TOKEN`` cannot create advisories, so the report job
    runs with ``permissions: {}`` and mints a short-lived token, scoped to
    the repository, from the organisation's GitHub App (*Repository
    security advisories: read and write*) with
    ``actions/create-github-app-token``. The App's client ID is the
    ``SECURITY_ADVISORY_APP_CLIENT_ID`` variable and its private key the
    ``SECURITY_ADVISORY_APP_PRIVATE_KEY`` secret; without them the job fails
    with an error naming both. Suppressions are never replaced by lowering
    a severity threshold.
  - ``publish-wiki.yml`` (on pushes to ``standard-coxph`` or ``main``, on
    tags and on manual dispatch) converts ``docs/`` with pandoc and mirrors
    it to the wiki. It publishes only the tip of the default branch, which
    it reads from the repository, so the file works unchanged in a fork: a
    version tag at that tip publishes the release, any other push the
    development version, and a banner on every page states which. A tag
    elsewhere leaves the wiki unchanged. The wiki must exist (one page
    created in the web interface) before the first run.
  - ``.github/dependabot.yml`` keeps pip, Docker (base-image digest) and the
    pinned action SHAs current; actions are pinned by commit SHA.

## Do not

- Do not change wire signatures without updating ``algorithm_store.json``
  **and** ``central`` in the same commit.
- Do not commit test data or local configuration with secrets.
- Do not add ``__init__.py`` author/license headers unless asked.
- Do not rename the ``v6-cox-ph`` package directory — the hyphen is part of
  the vantage6 module contract.
