"""
Docker-free end-to-end pipeline tests for the federated Cox-PH algorithm.

These tests drive ``central`` in-process through ``MockAlgorithmClient`` and
compare the federated output against ``lifelines.CoxPHFitter`` fitted on the
pooled data of the included organisations. They require no Docker or vantage6
network.

Exactness tests run with ``COXPH_MIN_RISK_SET_CHANGE=1`` (guards disabled) so
the federated result matches lifelines closely. A separate test exercises the
default guards (``k=5``) and asserts the risk-set privacy property on per-node
``agg1`` at ``beta = 0``.

The released aggregates are always quantised to float32 precision on the
wire (about 7 significant figures), so exactness is capped at that floor:
the tightest exactness tolerances sit just above it (1e-5 / 1e-6), far below
any real decomposition or assembly error.
"""

import sys
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from vantage6.algorithm.tools.exceptions import AlgorithmError, PrivacyThresholdViolation, UserInputError

# Add the algorithm module and repo root to the path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "v6-cox-ph"))

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient  # noqa: E402

# Import the central module through the (hyphenated) package so that its
# relative imports resolve correctly.
central_module = import_module("v6-cox-ph.central")  # noqa: E402


DATA_DIR = repo_root / "tests" / "data"
EXPL_VARS = ["age", "treatment"]


def _load_datasets():
    """Load the three test CSVs."""
    df1 = pd.read_csv(DATA_DIR / "coxph_test_data_1.csv")
    df2 = pd.read_csv(DATA_DIR / "coxph_test_data_2.csv")
    df3 = pd.read_csv(DATA_DIR / "coxph_test_data_3.csv")
    return df1, df2, df3


def _run_central(datasets, organization_ids, expl_vars=None):
    """Run ``central`` through the mock client and return the result dict."""
    client = MockAlgorithmClient(
        datasets=datasets,
        module="v6-cox-ph",
        organization_ids=organization_ids,
    )
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": EXPL_VARS if expl_vars is None else expl_vars,
                "organization_ids": organization_ids,
            },
        },
        organizations=[organization_ids[0]],
    )
    results = client.wait_for_results(task_id=task["id"])
    return results[0]


def _lifelines_reference(df, expl_vars):
    """Fit a centralised Cox-PH model with lifelines and return key outputs."""
    from lifelines import CoxPHFitter

    central_df = df[["time", "event"] + expl_vars].copy()
    central_df["event"] = central_df["event"].astype(bool)
    cph = CoxPHFitter()
    cph.fit(central_df, duration_col="time", event_col="event")
    return {
        "coef": cph.params_.to_dict(),
        "se": cph.standard_errors_.to_dict(),
        "aic": float(cph.AIC_partial_),
    }


def _simulate_survival_data(seed: int, n_rows: int = 200, n_nodes: int = 3) -> list:
    """Simulate survival datasets (fixed seed) for the guard-bias scenarios.

    ``n_nodes`` nodes of ``n_rows`` rows each; event times follow an
    exponential hazard with the study's true coefficients, censored
    uniformly. The guard scenarios need nodes well above the reconstruction
    bound so the guards approximate the fit rather than flatten it.
    """
    beta_true = np.array([0.5, -0.3])
    rng = np.random.default_rng(seed)
    datasets = []
    for _ in range(n_nodes):
        X = rng.normal(size=(n_rows, 2))
        eta = X @ beta_true
        hazard = 0.02 * np.exp(eta)
        event_time = rng.exponential(1.0 / hazard)
        censor_time = rng.uniform(0.0, 60.0)
        datasets.append(
            pd.DataFrame(
                {
                    "time": np.minimum(event_time, censor_time),
                    "event": (event_time <= censor_time).astype(int),
                    "x1": X[:, 0],
                    "x2": X[:, 1],
                }
            )
        )
    return datasets


@pytest.fixture
def guards_off(monkeypatch):
    """Disable the risk-set jump guard (k=1) and binning for exactness tests.

    COXPH_PRIVACY_LEVEL=minimal also switches the hardening bundle off
    (reconstruction row bound, small-count time suppression, aggregate
    rounding): the exactness tests must run with only the mandatory guards.
    """
    monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "1")
    monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "minimal")
    monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
    monkeypatch.delenv("COXPH_MIN_RISK_SET_SIZE", raising=False)


@pytest.fixture
def standard_result(guards_off):
    """Run the federated algorithm on nodes 1 and 2 (guards off).

    Node 3 has too few events; under the fail-closed threshold policy a run
    that includes it stops with ``PrivacyThresholdViolation``, so the
    happy-path fixtures select organisations 1 and 2 explicitly.
    """
    df1, df2, _ = _load_datasets()
    datasets = [
        [{"database": df1, "db_type": "csv"}],
        [{"database": df2, "db_type": "csv"}],
    ]
    return _run_central(datasets, [1, 2])


@pytest.mark.unit
class TestFederatedPipelineVsLifelines:
    """Compare the federated result (guards off) against a lifelines reference."""

    def test_run_including_below_threshold_node_fails(self, guards_off):
        """A run that includes node 3 (too few events) fails closed."""
        df1, df2, df3 = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
            [{"database": df3, "db_type": "csv"}],
        ]
        with pytest.raises(PrivacyThresholdViolation):
            _run_central(datasets, [1, 2, 3])

    def test_converged(self, standard_result):
        """The optimiser should report convergence within a few iterations."""
        assert standard_result["converged"] is True
        assert isinstance(standard_result["n_iterations"], int)
        assert standard_result["n_iterations"] <= 10

    def test_coefficients_match_lifelines(self, standard_result):
        """Federated coefficients within 2e-3 of lifelines on pooled data."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        model = pd.DataFrame(standard_result["model"]).T
        for var in EXPL_VARS:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 2e-3, (
                f"coef mismatch {var}: fed={model.loc[var, 'Coef']}, " f"ref={ref['coef'][var]}"
            )

    def test_se_match_lifelines(self, standard_result):
        """Federated SE within 1e-3 of lifelines."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        model = pd.DataFrame(standard_result["model"]).T
        for var in EXPL_VARS:
            assert abs(model.loc[var, "SE"] - ref["se"][var]) <= 1e-3, (
                f"SE mismatch {var}: fed={model.loc[var, 'SE']}, " f"ref={ref['se'][var]}"
            )

    def test_z_matches_beta_over_se(self, standard_result):
        """Regression: Z must equal Coef / SE (not the old (exp(beta)-1)/SE)."""
        model = pd.DataFrame(standard_result["model"]).T
        for var in EXPL_VARS:
            coef = model.loc[var, "Coef"]
            se = model.loc[var, "SE"]
            z = model.loc[var, "Z"]
            assert abs(z - coef / se) <= 1e-9, f"Z != Coef/SE for {var}: Z={z}, Coef/SE={coef / se}"

    def test_aic_close_to_lifelines(self, standard_result):
        """AIC within 0.1 of the lifelines AIC."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        assert (
            abs(standard_result["aic"] - ref["aic"]) <= 0.1
        ), f"AIC mismatch: fed={standard_result['aic']}, ref={ref['aic']}"


@pytest.mark.unit
class TestFederatedEqualsPooled:
    """The federated run must equal a centralised run on the pooled data."""

    def test_coefficients_identical_to_pooled(self, guards_off):
        """Federated (2 nodes) coefficients equal pooled single-node to 1e-6.

        The wire quantisation is not additive — each node's aggregates are
        quantised to float32 precision before central sums them, while the
        pooled node quantises the total — so the two runs agree only up to
        the wire precision (measured 9.6e-9 on this data; 1e-6 keeps
        headroom while still catching any real decomposition error).
        """
        df1, df2, _ = _load_datasets()

        fed = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )

        pooled_df = pd.concat([df1, df2], ignore_index=True)
        pooled = _run_central([[{"database": pooled_df, "db_type": "csv"}]], [1])

        fed_model = pd.DataFrame(fed["model"]).T
        pooled_model = pd.DataFrame(pooled["model"]).T
        for var in EXPL_VARS:
            np.testing.assert_allclose(
                fed_model.loc[var, "Coef"],
                pooled_model.loc[var, "Coef"],
                atol=1e-6,
                err_msg=f"coef differs from pooled for {var}",
            )


@pytest.mark.unit
class TestNonConvergence:
    """The output reports non-convergence when the epoch budget is exhausted."""

    def test_forced_non_convergence(self, monkeypatch, guards_off):
        """With a single epoch the optimiser cannot converge.

        The reported Coef is beta_0 (all zeros) because the final
        Newton step is not applied. SE, Z and AIC are evaluated at the
        same beta_0.
        """
        monkeypatch.setattr(central_module, "MAX_ITERATIONS", 1)

        df1, df2, _ = _load_datasets()
        result = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )

        assert result["converged"] is False
        assert result["n_iterations"] == 1
        warning_text = " ".join(result["warnings"])
        assert "did not converge" in warning_text
        assert "last evaluated beta" in warning_text
        # A model is still returned
        assert result["model"] is not None

        # Coef == 0 (beta_0 reported, not beta_1)
        model = pd.DataFrame(result["model"]).T
        for var in EXPL_VARS:
            assert (
                model.loc[var, "Coef"] == 0.0
            ), f"Non-converged Coef should be 0 (beta_0), got {model.loc[var, 'Coef']}"
            # Z = Coef / SE = 0 / SE = 0
            assert model.loc[var, "Z"] == 0.0
            # SE must be finite (computed from the Hessian at beta_0)
            assert np.isfinite(model.loc[var, "SE"])


@pytest.mark.unit
class TestSingularHessian:
    """A singular Hessian produces a graceful result, not a crash."""

    def test_collinear_covariate(self, monkeypatch, guards_off):
        """Collinear covariate -> converged=False, NaN SE, JSON-serialisable."""
        df1, df2, _ = _load_datasets()
        df1 = df1.copy()
        df2 = df2.copy()
        df1["age2"] = df1["age"]
        df2["age2"] = df2["age"]
        collinear_vars = ["age", "age2"]

        client = MockAlgorithmClient(
            datasets=[
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )
        task = client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": collinear_vars,
                    "organization_ids": [1, 2],
                },
            },
            organizations=[1],
        )
        results = client.wait_for_results(task_id=task["id"])
        result = results[0]

        assert result["converged"] is False
        model = pd.DataFrame(result["model"]).T
        for var in collinear_vars:
            assert np.isnan(model.loc[var, "SE"]), f"SE for {var} should be NaN"
        assert result["model"] is not None
        # The singular cause must appear in the warnings list
        warning_text = " ".join(result["warnings"])
        assert "singular" in warning_text.lower(), f"Expected 'singular' in warnings, got: {warning_text}"


@pytest.mark.unit
class TestValidateRisksetAggregates:
    """Structural sanity bounds on a node's released risk-set sizes."""

    def test_increasing_sizes_rejected(self):
        from importlib import import_module

        from vantage6.algorithm.tools.exceptions import AlgorithmError

        central = import_module("v6-cox-ph.central")
        with pytest.raises(AlgorithmError, match="Organisation 2"):
            central._validate_riskset_aggregates(2, [5.0, 3.0, 4.0])

    def test_negative_rejected(self):
        from importlib import import_module

        from vantage6.algorithm.tools.exceptions import AlgorithmError

        central = import_module("v6-cox-ph.central")
        with pytest.raises(AlgorithmError, match="Organisation 3"):
            central._validate_riskset_aggregates(3, [5.0, -1.0, 0.0])

    def test_zero_first_rejected(self):
        from importlib import import_module

        from vantage6.algorithm.tools.exceptions import AlgorithmError

        central = import_module("v6-cox-ph.central")
        with pytest.raises(AlgorithmError, match="Organisation 3"):
            central._validate_riskset_aggregates(3, [0.0, 0.0, 0.0])

    def test_valid_sequences_pass(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")

        # held releases and a shrinking tail, including the legitimate
        # zeros at pooled grid points beyond the node's own data
        central._validate_riskset_aggregates(1, [5.0, 5.0, 3.0, 1.0])
        central._validate_riskset_aggregates(1, [5.0, 3.0, 0.0, 0.0])


@pytest.mark.unit
class TestRequireAllOrganisationsAnswered:
    """The fail-closed aggregation error names the missing organisations."""

    def test_names_the_missing_organisation_ids(self):
        from importlib import import_module

        from vantage6.algorithm.tools.exceptions import AlgorithmError

        central = import_module("v6-cox-ph.central")
        results = [{"organization_id": 1}, {"organization_id": 2}]
        with pytest.raises(AlgorithmError, match="no result from organisation ids"):
            central._require_all_organisations_answered(results, [1, 2, 3])

    def test_missing_ids_are_listed(self):
        from importlib import import_module

        from vantage6.algorithm.tools.exceptions import AlgorithmError

        central = import_module("v6-cox-ph.central")
        results = [{"organization_id": 1}, {"organization_id": 3}]
        with pytest.raises(AlgorithmError) as excinfo:
            central._require_all_organisations_answered(results, [1, 2, 3])
        assert "[2]" in str(excinfo.value)

    def test_passes_when_all_answered(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")

        central._require_all_organisations_answered([{"organization_id": 1}, {"organization_id": 2}], [1, 2])


@pytest.mark.unit
class TestSubTaskValidation:
    """The result parser catches malformed sub-task results.

    Structural errors (missing keys) surface in pydantic's wording
    (``Field required``); the semantic checks (shape, finiteness, unknown
    organisation, missing variables) keep their custom wording.
    """

    def test_parse_iteration_result_bad_agg1_length(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="agg1 has length"):
            central._parse_result(
                central.IterationResult,
                {"organization_id": 1, "agg1": [1.0, 2.0], "agg2": [], "agg3": []},
                context={"expected_ids": [1], "n_times": 5, "n_covs": 2},
            )

    def test_parse_iteration_result_missing_key(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="Field required"):
            central._parse_result(
                central.IterationResult,
                {"organization_id": 1, "agg1": [1.0]},
                context={"expected_ids": [1], "n_times": 1, "n_covs": 1},
            )

    def test_parse_zsum_result_missing_sum(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="Field required"):
            central._parse_result(
                central.SummedZResult,
                {"organization_id": 1},
                context={"expected_ids": [1], "expl_vars": ["age", "treatment"], "time_col": "time"},
            )

    def test_parse_zsum_result_missing_variable(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="missing variables"):
            central._parse_result(
                central.SummedZResult,
                {
                    "organization_id": 1,
                    "sum": {"age": 10.0},
                    "sum_squares": {"age": 100.0, "treatment": 25.0},
                    "times": {"time": [1.0], "freq": [1]},
                    "privacy_settings": {"sample_size_threshold": 10},
                },
                context={"expected_ids": [1], "expl_vars": ["age", "treatment"], "time_col": "time"},
            )

    def test_parse_zsum_result_missing_times(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="Field required"):
            central._parse_result(
                central.SummedZResult,
                {
                    "organization_id": 1,
                    "sum": {"age": 10.0, "treatment": 5.0},
                    "sum_squares": {"age": 100.0, "treatment": 25.0},
                    "privacy_settings": {"sample_size_threshold": 10},
                },
                context={"expected_ids": [1], "expl_vars": ["age", "treatment"], "time_col": "time"},
            )

    def test_parse_result_unknown_organization(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="not one of"):
            central._parse_result(
                central.IterationResult,
                {"organization_id": 99, "agg1": [1.0], "agg2": [[1.0]], "agg3": [[[1.0]]]},
                context={"expected_ids": [1], "n_times": 1, "n_covs": 1},
            )

    def test_parse_result_not_a_dict(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="Expected a dict result"):
            central._parse_result(central.SummedZResult, [1, 2, 3], context={"expected_ids": [1]})


@pytest.mark.unit
class TestWireContractPinning:
    """The partials' result dicts carry exactly the wire-model keys.

    The partials build their results from ``SummedZResult`` /
    ``IterationResult`` and return ``model_dump()``; these tests pin the
    dumped key sets against literals so the wire contract is asserted next
    to ``algorithm_store.json``.
    """

    def _client(self):
        df1, df2, _ = _load_datasets()
        return MockAlgorithmClient(
            datasets=[
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )

    def test_compute_summed_z_result_keys(self, guards_off):
        client = self._client()
        task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": EXPL_VARS},
            },
            organizations=[1, 2],
        )
        results = client.wait_for_results(task_id=task["id"])
        assert results, "compute_summed_z returned no results"
        for r in results:
            assert set(r.keys()) == {"organization_id", "sum", "sum_squares", "times", "privacy_settings"}
            assert set(r["times"].keys()) == {"time", "freq"}
            assert set(r["privacy_settings"].keys()) == {
                "sample_size_threshold",
                "min_risk_set_change",
                "time_bin_width",
                "min_time_event_count",
                "aggregate_sig_figs",
                "min_risk_set_size",
            }

    def test_perform_iteration_result_keys(self, guards_off):
        client = self._client()
        grid_task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": EXPL_VARS},
            },
            organizations=[1, 2],
        )
        grid = sorted(
            {
                t
                for r in client.wait_for_results(task_id=grid_task["id"])
                for t in pd.DataFrame.from_dict(r["times"])["time"].tolist()
            }
        )
        task = client.task.create(
            input_={
                "method": "perform_iteration",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": EXPL_VARS,
                    "beta": [0.0, 0.0],
                    "centre": [0.0, 0.0],
                    "scale": [1.0, 1.0],
                    "unique_time_events": grid,
                },
            },
            organizations=[1, 2],
        )
        results = client.wait_for_results(task_id=task["id"])
        assert results, "perform_iteration returned no results"
        for r in results:
            assert set(r.keys()) == {"organization_id", "agg1", "agg2", "agg3"}


@pytest.mark.unit
class TestNaNPolicy:
    """NaN rows are dropped identically in all partials."""

    def test_nan_rows_dropped_consistently(self, monkeypatch, guards_off):
        """A NaN in age on an event row of node 1 is dropped in
        compute_summed_z (and thus excluded from event counts, z_sum and
        risk sets); the federated result matches lifelines on the NaN-free
        pooled data.

        Putting the NaN on an *event* row is critical: without the shared
        preparation step the event counts (freq) could include rows that
        z_sum and the risk sets exclude — a 10-20x tolerance violation.
        """
        df1, df2, _ = _load_datasets()
        df1 = df1.copy()
        # Introduce NaN in age on the first event row (row 1, event=1)
        event_rows = df1[df1["event"] == 1].index[:3].tolist()
        for idx in event_rows:
            df1.loc[idx, "age"] = np.nan

        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        result = _run_central(datasets, [1, 2])
        assert result["converged"] is True

        # Build the NaN-free reference
        df1_clean = df1.dropna(subset=["time", "event", "age", "treatment"])
        ref = _lifelines_reference(pd.concat([df1_clean, df2]), EXPL_VARS)
        model = pd.DataFrame(result["model"]).T
        for var in EXPL_VARS:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 2e-3, (
                f"coef mismatch {var}: fed={model.loc[var, 'Coef']}, " f"ref={ref['coef'][var]}"
            )


def _per_node_agg1_at_beta_zero(datasets, organization_ids, expl_vars=None):
    """Dispatch perform_iteration at beta=0 and return per-node agg1 lists.

    At beta=0, exp(beta . X) = 1, so agg1[t] equals the risk-set size |R(t)|.
    """
    expl_vars = EXPL_VARS if expl_vars is None else expl_vars
    client = MockAlgorithmClient(
        datasets=datasets,
        module="v6-cox-ph",
        organization_ids=organization_ids,
    )

    # Gather the pooled event-time grid from compute_summed_z results.
    task = client.task.create(
        input_={
            "method": "compute_summed_z",
            "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": expl_vars},
        },
        organizations=organization_ids,
    )
    time_results = client.wait_for_results(task_id=task["id"])
    grid = []
    for r in time_results:
        t = pd.DataFrame.from_dict(r["times"])
        grid.extend(t["time"].tolist())
    grid = sorted(set(grid))

    # Dispatch perform_iteration at beta = 0.
    task = client.task.create(
        input_={
            "method": "perform_iteration",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": expl_vars,
                "beta": [0.0, 0.0],
                "centre": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "unique_time_events": grid,
            },
        },
        organizations=organization_ids,
    )
    iter_results = client.wait_for_results(task_id=task["id"])
    return [r["agg1"] for r in iter_results]


@pytest.mark.unit
class TestDefaultGuardsPrivacyProperty:
    """With default guards (k=5), shared risk-set aggregates are protected."""

    def test_agg1_changes_at_least_k(self, monkeypatch):
        """Every positive decrease in per-node agg1 is >= k and min >= k.

        The default bundle also carries the absolute floor (standard level:
        R = 50): every released value is at least R, except values held at
        the first grid point, which is always released whatever its size.

        The scenario uses 200-row nodes: above the standard level's
        admission bound (max(10 x k=5, 3 x R=50) = 150 rows), so the nodes
        are admitted and the floor holds only the tail of each node's
        risk-set trajectory.
        """
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        k = 5
        r = 50
        expl_vars = ["x1", "x2"]
        datasets = [[{"database": df, "db_type": "csv"}] for df in _simulate_survival_data(20261005)]
        per_node_agg1 = _per_node_agg1_at_beta_zero(datasets, [1, 2, 3], expl_vars=expl_vars)

        for node_idx, agg1 in enumerate(per_node_agg1):
            values = [float(v) for v in agg1]
            # Non-increasing
            for i in range(1, len(values)):
                assert values[i] <= values[i - 1] + 1e-9, f"node {node_idx}: agg1 not non-increasing at {i}"
            # Every positive change (decrease) is >= k
            for i in range(1, len(values)):
                decrease = values[i - 1] - values[i]
                if decrease > 1e-9:
                    assert decrease >= k - 1e-9, f"node {node_idx}: decrease {decrease} < k={k} at {i}"
            # Smallest non-zero value is >= k
            non_zero = [v for v in values if v > 1e-9]
            assert non_zero, f"node {node_idx}: all agg1 are zero"
            assert min(non_zero) >= k - 1e-9, f"node {node_idx}: min non-zero agg1 {min(non_zero)} < k={k}"
            # Floor: every released value is >= R, except values held at the
            # always-released first grid point.
            for i, v in enumerate(values):
                if v > 1e-9:
                    assert v >= r - 1e-9 or abs(v - values[0]) <= 1e-9, (
                        f"node {node_idx}: agg1 {v} at {i} is below the floor R={r} "
                        f"and is not the held first grid point"
                    )

    def test_coefficients_within_loose_bound(self, monkeypatch):
        """With default guards, coefficients stay within 0.1 of lifelines.

        The scenario uses 150-row nodes — exactly the standard level's
        admission bound (max(10 x k=5, 3 x R=50) = 150 rows), so this is
        the smallest node size the default guards admit. There the default
        floor R=50 holds only the tail of each node's risk-set
        trajectory, so the guards approximate the fit. Smaller nodes are
        refused by the admission bound (TestAdmissionBound below).
        """
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        datasets = _simulate_survival_data(20261005, n_rows=150)
        result = _run_central(
            [[{"database": df, "db_type": "csv"}] for df in datasets],
            [1, 2, 3],
            expl_vars=["x1", "x2"],
        )
        assert result["converged"] is True
        assert result["privacy_guards"]["active"] is True
        ref = _lifelines_reference(pd.concat(datasets), ["x1", "x2"])
        model = pd.DataFrame(result["model"]).T
        for var in ["x1", "x2"]:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 0.1, (
                f"coef {var}: fed={model.loc[var, 'Coef']}, " f"ref={ref['coef'][var]}"
            )


@pytest.mark.unit
class TestTimeBinning:
    """With COXPH_TIME_BIN_WIDTH set, shared event times are on the grid."""

    def test_shared_times_on_grid(self, monkeypatch):
        """All shared event times are multiples of the bin width and it converges."""
        monkeypatch.setenv("COXPH_TIME_BIN_WIDTH", "10")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "1")

        df1, df2, _ = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        client = MockAlgorithmClient(
            datasets=datasets,
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )
        task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": EXPL_VARS,
                },
            },
            organizations=[1, 2],
        )
        results = client.wait_for_results(task_id=task["id"])
        shared_times = []
        for r in results:
            t = pd.DataFrame.from_dict(r["times"])
            shared_times.extend(t["time"].tolist())
        for t in shared_times:
            assert t % 10 == 0 or np.isclose(t % 10, 0), f"shared time {t} is not a multiple of 10"

        # The full pipeline still converges.
        result = _run_central(datasets, [1, 2])
        assert result["converged"] is True


def _run_central_with_client(client, organization_ids):
    """Dispatch central through a pre-built (possibly wrapped) client."""
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": EXPL_VARS,
                "organization_ids": organization_ids,
            },
        },
        organizations=[organization_ids[0]],
    )
    results = client.wait_for_results(task_id=task["id"])
    return results[0]


@pytest.mark.unit
class TestOrganizationAttribution:
    """Errors attribute results by their organization_id, not their position."""

    def _two_node_client(self):
        df1, df2, _ = _load_datasets()
        return MockAlgorithmClient(
            datasets=[
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )

    def test_partial_results_carry_organization_id(self, guards_off):
        """compute_summed_z and perform_iteration return their organisation id."""
        client = self._two_node_client()

        task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": EXPL_VARS,
                },
            },
            organizations=[1, 2],
        )
        results = client.wait_for_results(task_id=task["id"])
        assert sorted(r["organization_id"] for r in results) == [1, 2]

    def test_shuffled_results_attributed_correctly(self, guards_off, monkeypatch):
        """Reversed result order still yields the correct model."""
        original_wait = MockAlgorithmClient.wait_for_results

        def reversed_wait(self, task_id, interval=1):
            return list(reversed(original_wait(self, task_id, interval)))

        # Patch the class so the client copies the mock hands to central are
        # wrapped as well (central uses those for its sub-task waits).
        monkeypatch.setattr(MockAlgorithmClient, "wait_for_results", reversed_wait)

        client = self._two_node_client()
        result = _run_central_with_client(client, [1, 2])
        assert result["converged"] is True

        # The model must equal the unshuffled run
        reference = _run_central(
            [
                [{"database": _load_datasets()[0], "db_type": "csv"}],
                [{"database": _load_datasets()[1], "db_type": "csv"}],
            ],
            [1, 2],
        )
        shuffled_model = pd.DataFrame(result["model"]).T
        reference_model = pd.DataFrame(reference["model"]).T
        for var in EXPL_VARS:
            np.testing.assert_allclose(shuffled_model.loc[var, "Coef"], reference_model.loc[var, "Coef"], atol=1e-12)

    def test_unknown_organization_id_raises(self, guards_off, monkeypatch):
        """A result claiming an organisation that was not dispatched fails closed."""
        original_wait = MockAlgorithmClient.wait_for_results

        def tampered_wait(self, task_id, interval=1):
            results = original_wait(self, task_id, interval)
            for r in results:
                if isinstance(r, dict):
                    r["organization_id"] = 99
            return results

        monkeypatch.setattr(MockAlgorithmClient, "wait_for_results", tampered_wait)

        client = self._two_node_client()
        with pytest.raises(AlgorithmError, match="not one of"):
            _run_central_with_client(client, [1, 2])


@pytest.mark.unit
class TestAffineTransform:
    """The standardisation of the covariates must not change the model."""

    def _uncentred_fit(self, client, organization_ids):
        """Fit without centring/scaling: dispatch with centre 0 / scale 1."""
        from importlib import import_module

        coxph_logic = import_module("v6-cox-ph.coxph_logic")

        # Gather z_sum and the event grid
        task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": EXPL_VARS},
            },
            organizations=organization_ids,
        )
        results = client.wait_for_results(task_id=task["id"])
        z_sum = None
        time_event_dfs = []
        for r in results:
            if z_sum is None:
                z_sum = pd.Series(r["sum"])
            else:
                z_sum += pd.Series(r["sum"])
            time_event_dfs.append(pd.DataFrame.from_dict(r["times"]))
        ate = pd.concat(time_event_dfs).groupby("time", as_index=False).sum()
        grid = ate["time"].tolist()

        beta = np.zeros(len(EXPL_VARS))
        for _ in range(20):
            task = client.task.create(
                input_={
                    "method": "perform_iteration",
                    "kwargs": {
                        "time_col": "time",
                        "outcome_col": "event",
                        "expl_vars": EXPL_VARS,
                        "beta": beta.tolist(),
                        "centre": [0.0, 0.0],
                        "scale": [1.0, 1.0],
                        "unique_time_events": grid,
                    },
                },
                organizations=organization_ids,
            )
            results = client.wait_for_results(task_id=task["id"])
            n_times = len(grid)
            agg1 = np.zeros(n_times)
            agg2 = np.zeros((n_times, len(EXPL_VARS)))
            agg3 = np.zeros((n_times, len(EXPL_VARS), len(EXPL_VARS)))
            for r in results:
                agg1 += np.array(r["agg1"])
                agg2 += np.array(pd.DataFrame.from_dict(r["agg2"]))
                agg3 += np.array([np.array(lst) for lst in r["agg3"]])
            primary, secondary = coxph_logic.compute_derivatives(agg1, agg2, agg3, ate, z_sum)
            from scipy.linalg import solve

            step = solve(secondary, primary)
            if float(np.max(np.abs(step))) <= 1e-6:
                return beta
            beta = beta - step
        return beta

    def test_standardised_equals_unstandardised(self, guards_off):
        """central's standardised fit equals a centre-0/scale-1 fit to 1e-6.

        The affine invariance holds up to the wire precision: the two runs
        quantise differently scaled aggregates (float32 precision, always
        on), so their fits differ by the wire quantisation noise (measured
        1.1e-7 on this data).
        """
        df1, df2, _ = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]

        standardised = _run_central(datasets, [1, 2])
        assert standardised["converged"] is True

        client = MockAlgorithmClient(datasets=datasets, module="v6-cox-ph", organization_ids=[1, 2])
        unstandardised_beta = self._uncentred_fit(client, [1, 2])

        model = pd.DataFrame(standardised["model"]).T
        for i, var in enumerate(EXPL_VARS):
            np.testing.assert_allclose(
                model.loc[var, "Coef"], unstandardised_beta[i], atol=1e-6, err_msg=f"back-transform mismatch {var}"
            )

    @pytest.mark.parametrize("transform", ["scale_1e4", "offset_1e4"])
    def test_extreme_covariates_converge(self, guards_off, transform):
        """A covariate scaled by 1e4 or offset by 1e4 still converges."""
        df1, df2, _ = _load_datasets()
        df1 = df1.copy()
        df2 = df2.copy()
        if transform == "scale_1e4":
            df1["age"] = df1["age"] * 1e4
            df2["age"] = df2["age"] * 1e4
        else:
            df1["age"] = df1["age"] + 1e4
            df2["age"] = df2["age"] + 1e4

        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        result = _run_central(datasets, [1, 2])
        assert result["converged"] is True

        # The model must match lifelines on the same (extreme) data
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        model = pd.DataFrame(result["model"]).T
        for var in EXPL_VARS:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 2e-3, (
                f"coef mismatch {var} ({transform}): fed={model.loc[var, 'Coef']}, " f"ref={ref['coef'][var]}"
            )

    def test_missing_outcome_rows_equal_removal(self, guards_off):
        """Rows with a missing outcome give exactly the result of removing them."""
        df1, df2, _ = _load_datasets()
        df1 = df1.copy()
        nan_rows = df1.index[:4]
        df1.loc[nan_rows, "event"] = np.nan

        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        with_nan = _run_central(datasets, [1, 2])

        df1_dropped = df1.dropna(subset=["event"])
        removed = _run_central(
            [
                [{"database": df1_dropped, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )

        with_nan_model = pd.DataFrame(with_nan["model"]).T
        removed_model = pd.DataFrame(removed["model"]).T
        for var in EXPL_VARS:
            np.testing.assert_allclose(with_nan_model.loc[var, "Coef"], removed_model.loc[var, "Coef"], atol=1e-12)


def _breslow_log_likelihood(df, expl_vars, beta):
    """An independent Breslow partial log-likelihood on pooled data."""
    times = df["time"].to_numpy()
    events = df["event"].to_numpy()
    eta = df[expl_vars].to_numpy(dtype=float) @ np.asarray(beta, dtype=float)
    ll = 0.0
    for t in np.unique(times[events == 1]):
        event_mask = (times == t) & (events == 1)
        d_t = int(event_mask.sum())
        risk = times >= t
        s0 = float(np.exp(eta[risk]).sum())
        ll += float(eta[event_mask].sum()) - d_t * float(np.log(s0))
    return ll


@pytest.mark.unit
class TestOptimiser:
    """The central optimiser: log-likelihood, LR test and step-halving."""

    def test_log_likelihood_matches_direct_breslow(self, guards_off):
        """The reported log-likelihood equals a direct Breslow fit to 1e-5.

        The reported log-likelihood is computed from the wire-quantised
        aggregates (float32 precision, always on) while the direct fit uses
        full precision; the difference is the wire quantisation noise
        (measured 3.3e-7 on this data, bounded by events x 5e-7).
        """
        df1, df2, _ = _load_datasets()
        result = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )
        assert result["converged"] is True
        model = pd.DataFrame(result["model"]).T
        beta = model["Coef"].to_numpy(dtype=float)
        direct = _breslow_log_likelihood(pd.concat([df1, df2]), EXPL_VARS, beta)
        assert (
            abs(result["log_likelihood"] - direct) <= 1e-5
        ), f"log-likelihood mismatch: reported={result['log_likelihood']}, direct={direct}"

    def test_likelihood_ratio_fields_consistent(self, guards_off):
        """ll > ll_null, the LR statistic and p-value are consistent."""
        df1, df2, _ = _load_datasets()
        result = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )
        ll = result["log_likelihood"]
        ll_null = result["log_likelihood_null"]
        assert ll > ll_null
        assert result["n_events"] == int((pd.concat([df1, df2])["event"] == 1).sum())
        assert abs(result["lr_statistic"] - 2 * (ll - ll_null)) <= 1e-9
        assert 0.0 <= result["lr_p_value"] <= 1.0
        assert result["covariance"] is not None
        cov = np.asarray(result["covariance"], dtype=float)
        assert cov.shape == (len(EXPL_VARS), len(EXPL_VARS))
        assert result["algorithm_version"] not in ("", None)

    def test_step_halving_triggers(self, guards_off, monkeypatch):
        """An oversized Newton step is detected and halved."""
        original_solve = central_module.solve
        calls = {"n": 0}

        def overshooting_solve(a, b):
            step = original_solve(a, b)
            if calls["n"] == 0:
                # Triple the very first step: the next evaluation must show a
                # likelihood decrease and engage the step-halving.
                calls["n"] += 1
                return 3.0 * step
            return step

        monkeypatch.setattr(central_module, "solve", overshooting_solve)

        warnings_seen = []
        monkeypatch.setattr(central_module, "warn", lambda msg: warnings_seen.append(msg))

        df1, df2, _ = _load_datasets()
        result = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )
        assert any(
            "halving the step" in w for w in warnings_seen
        ), f"Expected step-halving warning, saw: {warnings_seen}"
        assert result["converged"] is True

        # The recovered model still matches lifelines
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        model = pd.DataFrame(result["model"]).T
        for var in EXPL_VARS:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 2e-3


@pytest.mark.unit
class TestSurvivalCurves:
    """Baseline cumulative hazard and survival curves (all computed centrally)."""

    def _run_central_with_profiles(self, datasets, organization_ids, profiles):
        client = MockAlgorithmClient(datasets=datasets, module="v6-cox-ph", organization_ids=organization_ids)
        task = client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": EXPL_VARS,
                    "organization_ids": organization_ids,
                    "covariate_profiles": profiles,
                },
            },
            organizations=[organization_ids[0]],
        )
        results = client.wait_for_results(task_id=task["id"])
        return results[0]

    def test_baseline_curve_consistency(self, guards_off):
        """S(t | centre) == exp(-H0(t)); curves only on the event grid."""
        df1, df2, _ = _load_datasets()
        result = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )
        h0 = result["baseline_cumulative_hazard"]
        baseline_curve = result["survival_curves"]["0"]
        assert set(h0.keys()) == set(baseline_curve.keys())
        for t in h0:
            assert abs(baseline_curve[t] - np.exp(-h0[t])) <= 1e-12
        # H0 is non-decreasing and starts at the first grid time
        values = [h0[t] for t in sorted(h0, key=float)]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))

    def test_baseline_hazard_matches_direct_breslow(self, guards_off):
        """H0 equals a direct Breslow computation at the same centre, to 1e-6.

        H0 is accumulated from the wire-quantised risk-set sums (float32
        precision, always on) while the direct computation uses full
        precision; the difference is the wire quantisation noise (measured
        3.0e-8 on this data).
        """
        from importlib import import_module

        coxph_logic = import_module("v6-cox-ph.coxph_logic")

        df1, df2, _ = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        result = _run_central(datasets, [1, 2])
        assert result["converged"] is True

        # Recompute the rounded centre/scale exactly as central does
        client = MockAlgorithmClient(datasets=datasets, module="v6-cox-ph", organization_ids=[1, 2])
        task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": EXPL_VARS},
            },
            organizations=[1, 2],
        )
        zsum_results = client.wait_for_results(task_id=task["id"])
        z_sum = None
        sum_squares = None
        for r in zsum_results:
            z_sum = pd.Series(r["sum"]) if z_sum is None else z_sum + pd.Series(r["sum"])
            sum_squares = (
                pd.Series(r["sum_squares"]) if sum_squares is None else sum_squares + pd.Series(r["sum_squares"])
            )
        n_events = float(result["n_events"])
        centre, scale = coxph_logic.pooled_standardisation(
            z_sum.to_numpy(dtype=float), sum_squares.to_numpy(dtype=float), n_events
        )

        # Direct Breslow baseline at x* = 0 (the covariate centre)
        pooled = pd.concat([df1, df2])
        beta_star = pd.DataFrame(result["model"]).T["Coef"].to_numpy(dtype=float) * scale
        x_star = (pooled[EXPL_VARS].to_numpy(dtype=float) - centre) / scale
        eta = x_star @ beta_star
        times = pooled["time"].to_numpy(dtype=float)
        events = pooled["event"].to_numpy(dtype=float)
        grid = np.array(sorted(float(t) for t in result["baseline_cumulative_hazard"]))
        direct = {}
        cumulative = 0.0
        for t in grid:
            risk = times >= t
            d_t = float(np.sum((times == t) & (events == 1)))
            if d_t > 0:
                cumulative += d_t / float(np.exp(eta[risk]).sum())
            direct[str(t)] = cumulative

        reported = result["baseline_cumulative_hazard"]
        for t in reported:
            assert abs(reported[t] - direct[t]) <= 1e-6, f"H0 mismatch at t={t}"

    def test_covariate_profiles(self, guards_off):
        """Each profile's curve is S0(t) ** exp(beta . (x - centre)/scale)."""
        profiles = [{"age": 70.0}, {"age": 40.0, "treatment": 1.0}]
        df1, df2, _ = _load_datasets()
        result = self._run_central_with_profiles(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
            profiles,
        )
        assert set(result["survival_curves"].keys()) == {"0", "1", "2"}
        baseline = result["survival_curves"]["0"]
        h0 = result["baseline_cumulative_hazard"]
        for idx, profile in enumerate(profiles, start=1):
            curve = result["survival_curves"][str(idx)]
            for t in h0:
                assert abs(curve[t] - baseline[t]) > 1e-9, f"profile {idx} equals the baseline at t={t}"

    def test_invalid_profile_rejected(self, guards_off):
        """A profile naming an unknown variable fails with UserInputError."""
        from vantage6.algorithm.tools.exceptions import UserInputError

        df1, df2, _ = _load_datasets()
        client = MockAlgorithmClient(
            datasets=[
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )
        with pytest.raises(UserInputError):
            client.task.create(
                input_={
                    "method": "central",
                    "kwargs": {
                        "time_col": "time",
                        "outcome_col": "event",
                        "expl_vars": EXPL_VARS,
                        "organization_ids": [1, 2],
                        "covariate_profiles": [{"nonexistent_var": 1.0}],
                    },
                },
                organizations=[1],
            )

    def test_curve_steps_respect_k(self, monkeypatch):
        """With default guards every curve time is backed by risk sets >= k.

        The scenario uses 200-row nodes, admitted under the standard
        level's admission bound (150 rows).
        """
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        k = 5
        expl_vars = ["x1", "x2"]
        datasets = [[{"database": df, "db_type": "csv"}] for df in _simulate_survival_data(20261005)]
        result = _run_central(datasets, [1, 2, 3], expl_vars=expl_vars)
        assert result["converged"] is True

        # At beta = 0 the per-node agg1 equals the node's risk-set size; the
        # guards guarantee every non-zero value is >= k.
        per_node_agg1 = _per_node_agg1_at_beta_zero(datasets, [1, 2, 3], expl_vars=expl_vars)
        grid = sorted(
            (float(t) for t in result["baseline_cumulative_hazard"]),
        )
        for node_idx, agg1 in enumerate(per_node_agg1):
            for i, value in enumerate(agg1):
                if float(value) > 0:
                    assert float(value) >= k - 1e-9, (
                        f"node {node_idx}: risk set {float(value)} < k={k} at grid index {i} "
                        f"(time {grid[i] if i < len(grid) else '?'})"
                    )

    def test_curve_time_keys_coarsened(self, monkeypatch, guards_off):
        """COXPH_CURVE_TIME_SIG_FIGS coarsens the reported curve time keys.

        An empty value reports the exact event times; 2 significant figures
        merge nearby times into one key, and a merged key carries the value
        at the largest original time.
        """
        coxph_logic = import_module("v6-cox-ph.coxph_logic")
        df1, df2, _ = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        monkeypatch.setenv("COXPH_CURVE_TIME_SIG_FIGS", "")
        exact = _run_central(datasets, [1, 2])
        monkeypatch.setenv("COXPH_CURVE_TIME_SIG_FIGS", "2")
        coarse = _run_central(datasets, [1, 2])

        assert exact["curve_time_sig_figs"] is None
        assert coarse["curve_time_sig_figs"] == 2

        # Expected coarsening of the exact run: ascending times, so the
        # later (larger) time overwrites the earlier one on a key collision.
        key_to_time = {}
        for t in sorted(exact["baseline_cumulative_hazard"], key=float):
            key = str(float(coxph_logic.round_sig(float(t), 2)))
            key_to_time[key] = t
        expected = {key: exact["baseline_cumulative_hazard"][t] for key, t in key_to_time.items()}
        assert coarse["baseline_cumulative_hazard"] == expected
        # Merging actually happened on this data (2 figures on 1-decimal times)
        assert len(expected) < len(exact["baseline_cumulative_hazard"])
        # The survival curves share the coarsened keys and values
        for profile, curve in coarse["survival_curves"].items():
            expected_curve = {key: exact["survival_curves"][profile][t] for key, t in key_to_time.items()}
            assert curve == expected_curve

    def test_curve_time_sig_figs_default_is_six(self, monkeypatch, guards_off):
        """The curve time keys are coarsened to 6 significant figures by default.

        The test data's event times carry one decimal, so 6 significant
        figures leave the keys exact; an empty value reports them exactly
        too, and the two runs agree.
        """
        df1, df2, _ = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        monkeypatch.delenv("COXPH_CURVE_TIME_SIG_FIGS", raising=False)
        result = _run_central(datasets, [1, 2])
        assert result["curve_time_sig_figs"] == 6

        monkeypatch.setenv("COXPH_CURVE_TIME_SIG_FIGS", "")
        exact = _run_central(datasets, [1, 2])
        assert exact["curve_time_sig_figs"] is None
        assert result["baseline_cumulative_hazard"] == exact["baseline_cumulative_hazard"]
        assert result["survival_curves"] == exact["survival_curves"]


@pytest.mark.unit
class TestPrivacyGuardsSummary:
    """The result carries an aggregate summary of the active guards."""

    def test_default_guards_summary(self, monkeypatch):
        """Default guards: active, k=5, no binning, warning appended.

        The scenario uses 200-row nodes, admitted under the standard
        level's admission bound (150 rows).
        """
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        datasets = _simulate_survival_data(20261005)
        result = _run_central(
            [[{"database": df, "db_type": "csv"}] for df in datasets],
            [1, 2, 3],
            expl_vars=["x1", "x2"],
        )
        summary = result["privacy_guards"]
        assert set(summary.keys()) == {
            "active",
            "max_min_risk_set_change",
            "time_binning",
            "max_min_time_event_count",
            "max_min_risk_set_size",
            "max_held_event_fraction",
        }
        assert summary["active"] is True
        assert summary["max_min_risk_set_change"] == 5
        assert summary["time_binning"] is False
        assert summary["max_min_time_event_count"] == 2
        assert summary["max_min_risk_set_size"] == 50
        held = summary["max_held_event_fraction"]
        assert 0.0 <= held <= 1.0
        assert any("guards were active" in w for w in result["warnings"])
        assert any("minimum risk-set size R=50" in w for w in result["warnings"])

    def test_guards_off_summary(self, guards_off):
        """Guards disabled: inactive, no warning."""
        df1, df2, _ = _load_datasets()
        result = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )
        summary = result["privacy_guards"]
        assert set(summary.keys()) == {
            "active",
            "max_min_risk_set_change",
            "time_binning",
            "max_min_time_event_count",
            "max_min_risk_set_size",
            "max_held_event_fraction",
        }
        assert summary["active"] is False
        assert summary["max_min_risk_set_change"] == 1
        assert summary["time_binning"] is False
        assert summary["max_min_time_event_count"] == 1
        assert summary["max_min_risk_set_size"] is None
        # Guards off (k=1): nothing can be held, so the fraction is 0.
        assert summary["max_held_event_fraction"] == 0.0
        assert not any("guards were active" in w for w in result["warnings"])

    def test_binning_summary(self, monkeypatch):
        """Time binning shows up as active."""
        monkeypatch.setenv("COXPH_TIME_BIN_WIDTH", "10")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "1")
        df1, df2, _ = _load_datasets()
        result = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )
        summary = result["privacy_guards"]
        assert set(summary.keys()) == {
            "active",
            "max_min_risk_set_change",
            "time_binning",
            "max_min_time_event_count",
            "max_min_risk_set_size",
            "max_held_event_fraction",
        }
        assert summary["active"] is True
        assert summary["time_binning"] is True
        assert summary["max_min_risk_set_change"] == 1
        assert summary["max_min_time_event_count"] == 2
        # The floor is configured (standard level) but k=1 disables it; the
        # summary reports the configuration, the warning does not name it.
        assert summary["max_min_risk_set_size"] == 50
        # k=1 disables the risk-set guards: nothing can be held.
        assert summary["max_held_event_fraction"] == 0.0
        assert not any("minimum risk-set size" in w for w in result["warnings"])


@pytest.mark.unit
class TestGuardBiasRegression:
    """Fixed-seed regression bound on the guards' coefficient bias.

    The bound is derived from the bias study (scripts/bias_study.py): the
    smoke run (2 repetitions, nodes of 200 and 1000 rows, k in {1, 5, 10},
    binning on/off) observed a maximum deviation from the k = 1 fit of
    ~0.04; the bound below carries headroom for the full 50-repetition
    study to re-derive it (do not loosen it without re-running the study).
    """

    BOUND = 0.05

    def _simulate(self, seed: int, n_rows: int = 200) -> list:
        return _simulate_survival_data(seed, n_rows=n_rows)

    def _run(self, datasets):
        client = MockAlgorithmClient(
            datasets=[[{"database": df, "db_type": "csv"}] for df in datasets],
            module="v6-cox-ph",
            organization_ids=[1, 2, 3],
        )
        task = client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": ["x1", "x2"],
                    "organization_ids": [1, 2, 3],
                },
            },
            organizations=[1],
        )
        return client.wait_for_results(task_id=task["id"])[0]

    def test_default_guards_stay_within_bound(self, monkeypatch):
        """With default guards the coefficients stay within the study bound."""
        datasets = self._simulate(20261005)
        covs = ["x1", "x2"]

        # The bound covers the jump guard's bias (the study varied k only).
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "1")
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        unguarded = self._run(datasets)
        assert unguarded["converged"] is True

        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        guarded = self._run(datasets)
        assert guarded["converged"] is True
        assert guarded["privacy_guards"]["active"] is True

        unguarded_model = pd.DataFrame(unguarded["model"]).T
        guarded_model = pd.DataFrame(guarded["model"]).T
        for var in covs:
            deviation = abs(guarded_model.loc[var, "Coef"] - unguarded_model.loc[var, "Coef"])
            assert (
                deviation <= self.BOUND
            ), f"Guard bias for {var} is {deviation:.4f}, above the regression bound {self.BOUND}"

    def test_standard_bundle_converges(self, monkeypatch):
        """The full standard bundle converges end-to-end on the study dataset."""
        monkeypatch.delenv("COXPH_PRIVACY_LEVEL", raising=False)
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.delenv("COXPH_MIN_TIME_EVENT_COUNT", raising=False)
        monkeypatch.delenv("COXPH_AGGREGATE_SIG_FIGS", raising=False)
        monkeypatch.delenv("COXPH_MIN_RISK_SET_SIZE", raising=False)
        datasets = self._simulate(20261005)
        result = self._run(datasets)
        assert result["converged"] is True


def _run_central_with_kwargs(datasets, client_org_ids, **kwargs):
    """Run ``central`` through the mock client with overridden input kwargs.

    ``client_org_ids`` are the organisations the mock collaboration holds
    data for; ``organization_ids`` in ``kwargs`` is what the task requests.
    """
    client = MockAlgorithmClient(
        datasets=datasets,
        module="v6-cox-ph",
        organization_ids=client_org_ids,
    )
    input_kwargs = {
        "time_col": "time",
        "outcome_col": "event",
        "expl_vars": EXPL_VARS,
    }
    input_kwargs.update(kwargs)
    if "organization_ids" not in input_kwargs:
        input_kwargs["organization_ids"] = client_org_ids
    task = client.task.create(
        input_={"method": "central", "kwargs": input_kwargs},
        organizations=[client_org_ids[0]],
    )
    results = client.wait_for_results(task_id=task["id"])
    return results[0]


@pytest.mark.unit
class TestCentralInputValidation:
    """central rejects semantically invalid input and oversized requests."""

    def _datasets(self):
        df1, df2, _ = _load_datasets()
        return [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]

    def test_duplicate_expl_vars_rejected(self):
        """Duplicate explanatory variables are rejected end-to-end."""
        with pytest.raises(UserInputError, match="duplicate variables"):
            _run_central_with_kwargs(self._datasets(), [1, 2], expl_vars=["age", "treatment", "age"])

    def test_time_col_equal_outcome_col_rejected(self):
        """time_col == outcome_col is rejected end-to-end."""
        with pytest.raises(UserInputError, match="different columns"):
            _run_central_with_kwargs(self._datasets(), [1, 2], outcome_col="time")

    def test_expl_var_naming_time_col_rejected(self):
        """An explanatory variable naming the time column is rejected."""
        with pytest.raises(UserInputError, match="time or outcome column"):
            _run_central_with_kwargs(self._datasets(), [1, 2], expl_vars=["time", "age"])

    def test_expl_var_naming_outcome_col_rejected(self):
        """An explanatory variable naming the outcome column is rejected."""
        with pytest.raises(UserInputError, match="time or outcome column"):
            _run_central_with_kwargs(self._datasets(), [1, 2], expl_vars=["age", "event"])

    def test_unknown_organization_id_rejected(self):
        """Organisation IDs absent from the collaboration are rejected."""
        with pytest.raises(UserInputError, match="Unknown organisation IDs"):
            _run_central_with_kwargs(self._datasets(), [1, 2], organization_ids=[1, 999])

    def test_too_many_expl_vars_rejected(self):
        """More explanatory variables than COXPH_MAX_COVARIATES is rejected."""
        expl_vars = ["age", "treatment"] + [f"extra_{i}" for i in range(19)]
        assert len(expl_vars) == 21
        with pytest.raises(UserInputError, match="Too many explanatory variables"):
            _run_central_with_kwargs(self._datasets(), [1, 2], expl_vars=expl_vars)

    def test_too_many_model_terms_rejected(self):
        """COXPH_MAX_COVARIATES counts model terms, not entries.

        11 three-category entries give 22 model terms, above the cap of 20,
        although only 11 entries were requested.
        """
        expl_vars = [
            {f"v{i}": {"reference": "a", "categories": {"a": ["a"], "b": ["b"], "c": ["c"]}}} for i in range(11)
        ]
        with pytest.raises(UserInputError, match="Too many explanatory variables"):
            _run_central_with_kwargs(self._datasets(), [1, 2], expl_vars=expl_vars)

    def test_event_time_grid_cap_rejected(self, monkeypatch, guards_off):
        """A pooled event-time grid beyond COXPH_MAX_GRID_POINTS is rejected.

        The error message names the configured cap and the remedies, never
        the actual grid size (that is data).
        """
        monkeypatch.setenv("COXPH_MAX_GRID_POINTS", "5")
        datasets = self._datasets()

        # The pooled grid central would build, for the message assertion.
        client = MockAlgorithmClient(
            datasets=datasets,
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )
        task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": EXPL_VARS},
            },
            organizations=[1, 2],
        )
        grid = set()
        for r in client.wait_for_results(task_id=task["id"]):
            grid.update(pd.DataFrame.from_dict(r["times"])["time"].tolist())
        assert len(grid) > 5

        with pytest.raises(UserInputError, match="COXPH_MAX_GRID_POINTS") as excinfo:
            _run_central_with_kwargs(datasets, [1, 2])
        assert str(len(grid)) not in str(excinfo.value)


@pytest.mark.unit
class TestReleasedAggregatePrecision:
    """The partials apply the configured precision to what they release."""

    def _client_and_grid(self):
        df1, df2, _ = _load_datasets()
        client = MockAlgorithmClient(
            datasets=[
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )
        task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": EXPL_VARS},
            },
            organizations=[1, 2],
        )
        grid = sorted(
            {
                t
                for r in client.wait_for_results(task_id=task["id"])
                for t in pd.DataFrame.from_dict(r["times"])["time"].tolist()
            }
        )
        return client, grid

    def test_perform_iteration_rounds_released_aggregates(self, monkeypatch, guards_off):
        """COXPH_AGGREGATE_SIG_FIGS rounds agg1/agg2/agg3 to that many figures."""
        coxph_logic = import_module("v6-cox-ph.coxph_logic")
        client, grid = self._client_and_grid()
        iter_kwargs = {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": EXPL_VARS,
            "beta": [0.3, -0.2],
            "centre": [0.1, -0.1],
            "scale": [1.0, 2.0],
            "unique_time_events": grid,
        }

        def dispatch():
            task = client.task.create(
                input_={"method": "perform_iteration", "kwargs": iter_kwargs},
                organizations=[1, 2],
            )
            results = client.wait_for_results(task_id=task["id"])
            return {r["organization_id"]: r for r in results}

        # An empty COXPH_AGGREGATE_SIG_FIGS disables the privacy rounding,
        # but the wire quantisation (float32 precision, always on) still
        # applies: every released value is a 7-significant-figure number.
        monkeypatch.setenv("COXPH_AGGREGATE_SIG_FIGS", "")
        raw = dispatch()
        monkeypatch.setenv("COXPH_AGGREGATE_SIG_FIGS", "3")
        rounded = dispatch()

        assert set(raw) == set(rounded) == {1, 2}
        for org_id in raw:
            for agg in ("agg1", "agg2", "agg3"):
                raw_arr = np.asarray(raw[org_id][agg], dtype=float)
                rounded_arr = np.asarray(rounded[org_id][agg], dtype=float)
                # The wire quantisation: released values are fixed points of
                # rounding to 7 significant figures.
                np.testing.assert_array_equal(raw_arr, coxph_logic.round_sig(raw_arr, 7))
                # The privacy rounding: released values equal their own
                # 3-significant-figure rounding, up to the 1-ulp wobble of
                # the multiply-round-divide round_sig — the wire quantisation
                # to 7 figures applied on top reaches the value through a
                # different division and can shift it by one ulp.
                np.testing.assert_allclose(rounded_arr, coxph_logic.round_sig(rounded_arr, 3), rtol=1e-15, atol=0.0)
                np.testing.assert_array_equal(rounded_arr, coxph_logic.round_sig(rounded_arr, 7))

    def test_released_aggregates_carry_wire_precision_in_minimal(self, monkeypatch):
        """The wire quantisation applies even when the privacy rounding is off.

        Under the minimal level (aggregate rounding off) the released
        aggregates are still quantised to float32 precision: every value
        equals its own 7-significant-figure rounding.
        """
        coxph_logic = import_module("v6-cox-ph.coxph_logic")
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "minimal")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "1")
        monkeypatch.setenv("COXPH_AGGREGATE_SIG_FIGS", "")
        client, grid = self._client_and_grid()
        task = client.task.create(
            input_={
                "method": "perform_iteration",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": EXPL_VARS,
                    "beta": [0.3, -0.2],
                    "centre": [0.1, -0.1],
                    "scale": [1.0, 2.0],
                    "unique_time_events": grid,
                },
            },
            organizations=[1, 2],
        )
        results = client.wait_for_results(task_id=task["id"])
        assert results, "perform_iteration returned no results"
        for r in results:
            for agg in ("agg1", "agg2", "agg3"):
                arr = np.asarray(r[agg], dtype=float)
                np.testing.assert_array_equal(arr, coxph_logic.round_sig(arr, 7))

    def test_small_time_counts_suppressed(self, monkeypatch):
        """compute_summed_z merges per-time event counts below the minimum."""
        monkeypatch.delenv("COXPH_PRIVACY_LEVEL", raising=False)
        monkeypatch.delenv("COXPH_MIN_TIME_EVENT_COUNT", raising=False)
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        # The risk-set floor is switched off: this test isolates the
        # small-count suppression, so the 60-row node only has to meet the
        # reconstruction bound (10 x k=5 = 50 rows).
        monkeypatch.setenv("COXPH_MIN_RISK_SET_SIZE", "")
        # 60 rows: an event at each of the 54 smallest distinct times (every
        # per-time count is 1, below the standard-level minimum of 2) and a
        # censored row at each of the 6 largest times, so the tail censoring
        # of the shared preparation leaves the event rows untouched.
        n_rows = 60
        n_events = 54
        df = pd.DataFrame(
            {
                "time": [float(t) for t in range(1, n_rows + 1)],
                "event": [1] * n_events + [0] * (n_rows - n_events),
                "x1": np.linspace(-1.0, 1.0, n_rows),
            }
        )
        client = MockAlgorithmClient(
            datasets=[[{"database": df, "db_type": "csv"}]],
            module="v6-cox-ph",
            organization_ids=[1],
        )
        task = client.task.create(
            input_={
                "method": "compute_summed_z",
                "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": ["x1"]},
            },
            organizations=[1],
        )
        results = client.wait_for_results(task_id=task["id"])
        assert len(results) == 1
        times = pd.DataFrame.from_dict(results[0]["times"])

        # No released per-time count is below the minimum, the event total is
        # preserved, and every merged time is one of the original times.
        assert (times["freq"] >= 2).all()
        assert int(times["freq"].sum()) == int(df["event"].sum())
        assert set(times["time"].tolist()) <= set(df["time"].tolist())
        assert len(times) < n_events


@pytest.mark.unit
class TestAdmissionBound:
    """Nodes below the admission bound are refused (fail closed).

    The standard level admits a node only above max(factor * k, 3 x R) =
    max(10 x 5, 3 x 50) = 150 rows: a smaller node would have (nearly)
    its whole content inside the smallest released risk set, which the
    floor alone cannot hide.
    """

    def test_small_nodes_refused(self, monkeypatch):
        """50- and 100-row nodes under the default guards fail closed."""
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        for n_rows in (50, 100):
            datasets = _simulate_survival_data(20261005, n_rows=n_rows)
            with pytest.raises(PrivacyThresholdViolation):
                _run_central(
                    [[{"database": df, "db_type": "csv"}] for df in datasets],
                    [1, 2, 3],
                    expl_vars=["x1", "x2"],
                )

    def test_admitted_nodes_held_fraction_below_warn_threshold(self, monkeypatch):
        """200-row nodes: the held-event fraction stays below the warn threshold."""
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        datasets = _simulate_survival_data(20261005)
        result = _run_central(
            [[{"database": df, "db_type": "csv"}] for df in datasets],
            [1, 2, 3],
            expl_vars=["x1", "x2"],
        )
        held = result["privacy_guards"]["max_held_event_fraction"]
        assert held == pytest.approx(0.05825242718446602)
        assert held < 0.25
        assert not any("held" in w.lower() for w in result["warnings"])


@pytest.mark.unit
class TestHeldEventFractionAbort:
    """Central aborts when the held-event fraction exceeds the configured cap.

    The construction: 185 rows censored at time 1 and 15 events at times
    2..16 on a single 200-row node (above the admission bound). The tail
    censoring (k=5) removes the four latest events, the small-count
    suppression merges the remaining singletons into five grid points,
    and every risk set is below the floor R=50, so the floor holds the
    whole trajectory at the first release: 9 of the 11 remaining events
    are scored against the held (unchanged) risk set.
    """

    @staticmethod
    def _node():
        return pd.DataFrame(
            {
                "time": [1.0] * 185 + [float(t) for t in range(2, 17)],
                "event": [0] * 185 + [1] * 15,
                "x1": np.linspace(-1.0, 1.0, 200),
            }
        )

    def test_aborts_at_default_cap(self, monkeypatch):
        """A held-event fraction above the cap raises UserInputError."""
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.delenv("COXPH_MAX_HELD_EVENT_FRACTION", raising=False)
        with pytest.raises(UserInputError, match=r"exceeds the configured cap 0\.5 \(COXPH_MAX_HELD_EVENT_FRACTION\)"):
            _run_central([[{"database": self._node(), "db_type": "csv"}]], [1], expl_vars=["x1"])

    def test_cap_disabled_completes_and_reports(self, monkeypatch):
        """With the abort disabled the run completes and reports the fraction."""
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.setenv("COXPH_MAX_HELD_EVENT_FRACTION", "")
        result = _run_central([[{"database": self._node(), "db_type": "csv"}]], [1], expl_vars=["x1"])
        assert result["privacy_guards"]["max_held_event_fraction"] == pytest.approx(9 / 11)
        assert any("held" in w.lower() and "0.818" in w for w in result["warnings"])

    def test_below_cap_reports_diagnostic(self, monkeypatch):
        """A held fraction in (0.25, 0.5) warns but does not abort.

        185 rows censored at time 1, 12 events at time 2 and one event at
        each of times 3..12: the tail censoring keeps 18 events, the
        suppression releases them at time 2 (12 events) and at times 4, 6
        and 8 (2 each), and every risk set is below the floor, so the
        release stays at the first grid point: 6 of the 18 events are
        scored against the held risk set.
        """
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.delenv("COXPH_MAX_HELD_EVENT_FRACTION", raising=False)
        df = pd.DataFrame(
            {
                "time": [1.0] * 185 + [2.0] * 12 + [float(t) for t in range(3, 13)],
                "event": [0] * 185 + [1] * 22,
                "x1": np.linspace(-1.0, 1.0, 207),
            }
        )
        result = _run_central([[{"database": df, "db_type": "csv"}]], [1], expl_vars=["x1"])
        assert result["privacy_guards"]["max_held_event_fraction"] == pytest.approx(6 / 18)
        assert any("held" in w.lower() and "0.333" in w for w in result["warnings"])


@pytest.mark.unit
class TestStrictPrivacyLevel:
    """The strict bundle converges on 400-row nodes (the scaled tolerance)."""

    def test_strict_converges_within_budget(self, monkeypatch):
        """strict on 400-row nodes: converged within 20 iterations, coefficients within 0.1."""
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "strict")
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.delenv("COXPH_MIN_TIME_EVENT_COUNT", raising=False)
        monkeypatch.delenv("COXPH_AGGREGATE_SIG_FIGS", raising=False)
        monkeypatch.delenv("COXPH_MIN_RISK_SET_SIZE", raising=False)
        monkeypatch.delenv("COXPH_RECONSTRUCTION_ROW_FACTOR", raising=False)
        datasets = _simulate_survival_data(20261005, n_rows=400)
        result = _run_central(
            [[{"database": df, "db_type": "csv"}] for df in datasets],
            [1, 2, 3],
            expl_vars=["x1", "x2"],
        )
        assert result["converged"] is True
        assert result["n_iterations"] <= 20
        # The strict bundle's 4-significant-figure aggregates cannot support
        # the primary step tolerance: convergence comes from the secondary
        # (stability) tolerance.
        assert result["convergence_tolerance"] == pytest.approx(central_module.STABLE_STEP_TOLERANCE)
        ref = _lifelines_reference(pd.concat(datasets), ["x1", "x2"])
        model = pd.DataFrame(result["model"]).T
        for var in ["x1", "x2"]:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 0.1, (
                f"coef {var}: fed={model.loc[var, 'Coef']}, " f"ref={ref['coef'][var]}"
            )


def _move_event_times_up(df, time_col, outcome_col, min_count):
    """Move event rows' times up to their merged time (test-side reference).

    An independent reimplementation of the suppression merge: per-time
    event counts accumulate until they reach ``min_count``; the group is
    emitted at its largest time and its event rows move there. A trailing
    group below the minimum merges backward into the previous emitted
    group; when nothing was emitted the whole grid is one group at the
    last time. Censored rows keep their time.
    """
    counts = df[df[outcome_col] == 1][time_col].value_counts().sort_index()
    times = counts.index.tolist()
    move_to = {}
    group_count = 0
    group = []
    emitted = None
    for t, f in zip(times, counts.to_numpy().tolist()):
        group.append(t)
        group_count += int(f)
        if group_count >= min_count:
            for gt in group:
                move_to[gt] = t
            emitted = t
            group = []
            group_count = 0
    for gt in group:
        move_to[gt] = emitted if emitted is not None else times[-1]
    moved = df.copy()
    event_mask = moved[outcome_col] == 1
    moved.loc[event_mask, time_col] = moved.loc[event_mask, time_col].map(move_to)
    return moved


@pytest.mark.unit
class TestSuppressionConsistency:
    """The suppression move equals a run on explicitly right-moved data.

    Run A analyses the original data with the small-count suppression on
    (standard level, k=1 so no risk-set guard interferes): perform_iteration
    moves the event rows' times up to their merged time. Run B analyses
    data the test moved explicitly the same way, with the suppression off
    (COXPH_MIN_TIME_EVENT_COUNT=1). Both runs must agree on every result
    field except the guards summary and the warnings, which describe the
    active guards rather than the fit.
    """

    def test_move_matches_explicitly_moved_data(self, monkeypatch):
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "1")
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.delenv("COXPH_PRIVACY_LEVEL", raising=False)
        monkeypatch.delenv("COXPH_MIN_TIME_EVENT_COUNT", raising=False)
        datasets = _simulate_survival_data(20261005)
        moved = [_move_event_times_up(df, "time", "event", 2) for df in datasets]

        suppressed = _run_central(
            [[{"database": df, "db_type": "csv"}] for df in datasets],
            [1, 2, 3],
            expl_vars=["x1", "x2"],
        )
        monkeypatch.setenv("COXPH_MIN_TIME_EVENT_COUNT", "1")
        explicit = _run_central(
            [[{"database": df, "db_type": "csv"}] for df in moved],
            [1, 2, 3],
            expl_vars=["x1", "x2"],
        )

        assert set(suppressed) == set(explicit)
        for key in suppressed:
            if key in ("warnings", "privacy_guards"):
                continue
            assert suppressed[key] == explicit[key], f"result key {key!r} differs"


SEX_CATEGORICAL = {"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}
SEX_SCORED = {"sex": {"scores": {"M": 0, "F": 1}}}


def _csv_datasets():
    """The CSV test data of nodes 1 and 2, as the mock collaboration holds them."""
    df1, df2, _ = _load_datasets()
    return [
        [{"database": df1, "db_type": "csv"}],
        [{"database": df2, "db_type": "csv"}],
    ]


def _datasets_with_dummy():
    """The CSV test data with a precomputed 0/1 female column."""
    df1, df2, _ = _load_datasets()
    for df in (df1, df2):
        df["sex_female"] = (df["sex"] == "F").astype(float)
    return [
        [{"database": df1, "db_type": "csv"}],
        [{"database": df2, "db_type": "csv"}],
    ]


@pytest.mark.unit
class TestCategoricalCovariates:
    """Categorical and scored covariates equal their precomputed encodings.

    In the full CSVs the male events number 14 on node 1 and 10 on node 2,
    so at the default threshold of 10 node 2 fails the category threshold
    (a ready-made "at the threshold" case) while a threshold of 9 passes.
    Node 3 already fails the native threshold.
    """

    @pytest.fixture
    def threshold_9(self, monkeypatch, guards_off):
        """Threshold 9: node 2's male category (10 events) passes it."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "9")

    @staticmethod
    def _assert_models_equal(result, reference, term, ref_term):
        """The two runs agree to 1e-9 on every model field."""
        model = pd.DataFrame(result["model"]).T
        ref_model = pd.DataFrame(reference["model"]).T
        assert list(model.index) == ["age", term]
        assert list(ref_model.index) == ["age", ref_term]
        for column in ("Coef", "Exp(coef)", "SE", "lower_CI", "upper_CI", "Z", "p-value"):
            np.testing.assert_allclose(
                model[column].to_numpy(dtype=float),
                ref_model[column].to_numpy(dtype=float),
                atol=1e-9,
                err_msg=f"{column} differs from the precomputed column",
            )
        np.testing.assert_allclose(result["covariance"], reference["covariance"], atol=1e-9)
        assert result["log_likelihood"] == pytest.approx(reference["log_likelihood"], abs=1e-9)
        assert result["n_events"] == reference["n_events"]

    def test_categorical_equals_precomputed_dummy(self, threshold_9):
        """The categorical sex covariate equals a run on a precomputed 0/1 column."""
        result = _run_central(_csv_datasets(), [1, 2], expl_vars=["age", SEX_CATEGORICAL])
        assert result["converged"] is True
        reference = _run_central(_datasets_with_dummy(), [1, 2], expl_vars=["age", "sex_female"])
        self._assert_models_equal(result, reference, "sex[F]", "sex_female")

    def test_scored_equals_precomputed_numeric(self, threshold_9):
        """A scored variable equals a run on its precomputed numeric column."""
        result = _run_central(_csv_datasets(), [1, 2], expl_vars=["age", SEX_SCORED])
        assert result["converged"] is True
        reference = _run_central(_datasets_with_dummy(), [1, 2], expl_vars=["age", "sex_female"])
        self._assert_models_equal(result, reference, "sex", "sex_female")

    def test_profile_label_equals_profile_value(self, threshold_9):
        """The profile {'sex': 'F'} equals the profile with the 0/1 column at 1."""
        client = MockAlgorithmClient(datasets=_csv_datasets(), module="v6-cox-ph", organization_ids=[1, 2])
        task = client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": ["age", SEX_CATEGORICAL],
                    "organization_ids": [1, 2],
                    "covariate_profiles": [{"sex": "F"}],
                },
            },
            organizations=[1],
        )
        result = client.wait_for_results(task_id=task["id"])[0]

        ref_client = MockAlgorithmClient(datasets=_datasets_with_dummy(), module="v6-cox-ph", organization_ids=[1, 2])
        ref_task = ref_client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": ["age", "sex_female"],
                    "organization_ids": [1, 2],
                    "covariate_profiles": [{"sex_female": 1.0}],
                },
            },
            organizations=[1],
        )
        reference = ref_client.wait_for_results(task_id=ref_task["id"])[0]

        assert result["survival_curves"] == reference["survival_curves"]
        assert result["baseline_cumulative_hazard"] == reference["baseline_cumulative_hazard"]

    def test_lifelines_with_dummy(self, threshold_9):
        """The categorical fit matches lifelines with a dummy column."""
        result = _run_central(_csv_datasets(), [1, 2], expl_vars=["age", SEX_CATEGORICAL])
        df1, df2, _ = _load_datasets()
        pooled = pd.concat([df1, df2])
        pooled["sex_female"] = (pooled["sex"] == "F").astype(float)
        ref = _lifelines_reference(pooled, ["age", "sex_female"])
        model = pd.DataFrame(result["model"]).T
        for var, ref_var in (("age", "age"), ("sex[F]", "sex_female")):
            assert (
                abs(model.loc[var, "Coef"] - ref["coef"][ref_var]) <= 2e-3
            ), f"coef mismatch {var}: fed={model.loc[var, 'Coef']}, ref={ref['coef'][ref_var]}"
            assert (
                abs(model.loc[var, "SE"] - ref["se"][ref_var]) <= 1e-3
            ), f"SE mismatch {var}: fed={model.loc[var, 'SE']}, ref={ref['se'][ref_var]}"

    def test_category_at_the_threshold_fails(self, guards_off):
        """At the default threshold 10, node 2's male category (10 events) fails."""
        with pytest.raises(PrivacyThresholdViolation):
            _run_central(_csv_datasets(), [1, 2], expl_vars=["age", SEX_CATEGORICAL])

    def test_category_above_the_threshold_passes(self, threshold_9):
        """At threshold 9 the same configuration passes and converges."""
        result = _run_central(_csv_datasets(), [1, 2], expl_vars=["age", SEX_CATEGORICAL])
        assert result["converged"] is True
        assert list(pd.DataFrame(result["model"]).T.index) == ["age", "sex[F]"]
