"""
Unit tests for the privacy guards module.

Covers settings parsing, the sample-size threshold, the parent-task guard,
and iteration-input validation. The time-binning / tail-censoring / jump-guard
functions are tested here as well (added in a later step). The categorical
covariate encoding, the category threshold and the unlisted-code handling
are tested here too.
"""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest
import jwt

from vantage6.algorithm.tools.exceptions import (
    AlgorithmError,
    PrivacyThresholdViolation,
    PrivacyViolation,
    UserInputError,
)

# The guards module has relative imports (the Pydantic models in
# miscellaneous.py), so it must be imported through the installed package,
# never with a sys.path hack + bare-name import.
privacy_guards = import_module("v6-cox-ph.privacy_guards")
miscellaneous = import_module("v6-cox-ph.miscellaneous")

DEFAULT_MIN_RISK_SET_CHANGE = privacy_guards.DEFAULT_MIN_RISK_SET_CHANGE
DEFAULT_SAMPLE_SIZE_THRESHOLD = privacy_guards.DEFAULT_SAMPLE_SIZE_THRESHOLD
PrivacySettings = privacy_guards.PrivacySettings
ResourceCaps = privacy_guards.ResourceCaps
bin_times = privacy_guards.bin_times
check_category_sizes = privacy_guards.check_category_sizes
check_sample_size = privacy_guards.check_sample_size
drop_incomplete_rows = privacy_guards.drop_incomplete_rows
encode_covariates = privacy_guards.encode_covariates
ensure_spawned_by_central = privacy_guards.ensure_spawned_by_central
event_time_counts = privacy_guards.event_time_counts
guarded_risk_set_aggregates = privacy_guards.guarded_risk_set_aggregates
load_curve_time_sig_figs = privacy_guards.load_curve_time_sig_figs
load_max_held_event_fraction = privacy_guards.load_max_held_event_fraction
load_privacy_settings = privacy_guards.load_privacy_settings
load_resource_caps = privacy_guards.load_resource_caps
merged_time_map = privacy_guards.merged_time_map
prepare_node_data = privacy_guards.prepare_node_data
prepare_time_column = privacy_guards.prepare_time_column
quantise_wire_aggregates = privacy_guards.quantise_wire_aggregates
resolve_expl_vars = privacy_guards.resolve_expl_vars
round_released_aggregates = privacy_guards.round_released_aggregates
suppress_small_time_counts = privacy_guards.suppress_small_time_counts
tail_cutoff = privacy_guards.tail_cutoff
validate_expl_vars = privacy_guards.validate_expl_vars
validate_iteration_input = privacy_guards.validate_iteration_input
validate_survival_columns = privacy_guards.validate_survival_columns
code_key = miscellaneous.code_key
from tests.unit.reference_risk_sets import guarded_risk_set_masks  # noqa: E402


@pytest.mark.unit
class TestLoadPrivacySettings:
    """Tests for load_privacy_settings."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Remove every privacy env var so the defaults / presets apply."""
        for var in [
            "SAMPLE_SIZE_THRESHOLD",
            "COXPH_TIME_BIN_WIDTH",
            "COXPH_MIN_RISK_SET_CHANGE",
            "COXPH_MIN_RISK_SET_SIZE",
            "COXPH_PRIVACY_LEVEL",
            "COXPH_RECONSTRUCTION_ROW_FACTOR",
            "COXPH_MIN_TIME_EVENT_COUNT",
            "COXPH_AGGREGATE_SIG_FIGS",
        ]:
            monkeypatch.delenv(var, raising=False)

    def test_defaults(self, clean_env):
        """Default settings when no env vars are set."""
        s = load_privacy_settings()
        assert s.sample_size_threshold == DEFAULT_SAMPLE_SIZE_THRESHOLD
        assert s.min_risk_set_change == DEFAULT_MIN_RISK_SET_CHANGE
        assert s.time_bin_width is None
        # The default privacy level is "standard" with its hardening bundle.
        assert s.privacy_level == "standard"
        assert s.reconstruction_row_factor == 10
        assert s.min_time_event_count == 2
        assert s.aggregate_sig_figs == 6
        assert s.min_risk_set_size == 50

    def test_custom_values(self, monkeypatch):
        """Custom values from env vars."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "15")
        monkeypatch.setenv("COXPH_TIME_BIN_WIDTH", "10")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "8")
        s = load_privacy_settings()
        assert s.sample_size_threshold == 15
        assert s.min_risk_set_change == 8
        assert s.time_bin_width == 10.0

    def test_level_strict(self, clean_env, monkeypatch):
        """The strict level prescribes the strict hardening bundle."""
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "strict")
        s = load_privacy_settings()
        assert s.privacy_level == "strict"
        assert s.reconstruction_row_factor == 20
        assert s.min_time_event_count == 5
        assert s.aggregate_sig_figs == 4
        assert s.min_risk_set_size == 100

    def test_level_minimal(self, clean_env, monkeypatch):
        """The minimal level turns the optional hardening off."""
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "minimal")
        s = load_privacy_settings()
        assert s.privacy_level == "minimal"
        assert s.reconstruction_row_factor == 1
        assert s.min_time_event_count == 1
        assert s.aggregate_sig_figs is None
        assert s.min_risk_set_size is None

    def test_level_case_insensitive(self, clean_env, monkeypatch):
        """The level is matched case-insensitively (and trimmed)."""
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "  StRiCt ")
        s = load_privacy_settings()
        assert s.privacy_level == "strict"
        assert s.reconstruction_row_factor == 20

    def test_invalid_level(self, clean_env, monkeypatch):
        """An unknown level raises UserInputError naming the allowed values."""
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "paranoid")
        with pytest.raises(UserInputError, match="COXPH_PRIVACY_LEVEL"):
            load_privacy_settings()

    def test_explicit_knob_beats_level(self, clean_env, monkeypatch):
        """A set override beats the level preset; unset knobs keep it."""
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "strict")
        monkeypatch.setenv("COXPH_RECONSTRUCTION_ROW_FACTOR", "7")
        monkeypatch.setenv("COXPH_MIN_TIME_EVENT_COUNT", "3")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_SIZE", "30")
        s = load_privacy_settings()
        assert s.reconstruction_row_factor == 7
        assert s.min_time_event_count == 3
        assert s.aggregate_sig_figs == 4
        assert s.min_risk_set_size == 30

    def test_empty_sig_figs_disables_rounding(self, clean_env, monkeypatch):
        """An empty COXPH_AGGREGATE_SIG_FIGS switches the rounding off."""
        monkeypatch.setenv("COXPH_AGGREGATE_SIG_FIGS", "")
        s = load_privacy_settings()
        assert s.aggregate_sig_figs is None

    def test_empty_min_risk_set_size_disables_floor(self, clean_env, monkeypatch):
        """An empty COXPH_MIN_RISK_SET_SIZE switches the floor off."""
        monkeypatch.setenv("COXPH_MIN_RISK_SET_SIZE", "")
        s = load_privacy_settings()
        assert s.min_risk_set_size is None

    def test_invalid_knob_values(self, clean_env, monkeypatch):
        """Invalid override values raise UserInputError naming the variable."""
        monkeypatch.setenv("COXPH_RECONSTRUCTION_ROW_FACTOR", "0")
        with pytest.raises(UserInputError, match="COXPH_RECONSTRUCTION_ROW_FACTOR"):
            load_privacy_settings()
        monkeypatch.setenv("COXPH_RECONSTRUCTION_ROW_FACTOR", "wide")
        with pytest.raises(UserInputError, match="COXPH_RECONSTRUCTION_ROW_FACTOR"):
            load_privacy_settings()
        monkeypatch.delenv("COXPH_RECONSTRUCTION_ROW_FACTOR")
        monkeypatch.setenv("COXPH_MIN_TIME_EVENT_COUNT", "-2")
        with pytest.raises(UserInputError, match="COXPH_MIN_TIME_EVENT_COUNT"):
            load_privacy_settings()
        monkeypatch.delenv("COXPH_MIN_TIME_EVENT_COUNT")
        monkeypatch.setenv("COXPH_AGGREGATE_SIG_FIGS", "0")
        with pytest.raises(UserInputError, match="COXPH_AGGREGATE_SIG_FIGS"):
            load_privacy_settings()
        monkeypatch.delenv("COXPH_AGGREGATE_SIG_FIGS")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_SIZE", "0")
        with pytest.raises(UserInputError, match="COXPH_MIN_RISK_SET_SIZE"):
            load_privacy_settings()
        monkeypatch.setenv("COXPH_MIN_RISK_SET_SIZE", "small")
        with pytest.raises(UserInputError, match="COXPH_MIN_RISK_SET_SIZE"):
            load_privacy_settings()
        monkeypatch.delenv("COXPH_MIN_RISK_SET_SIZE")

    def test_invalid_threshold(self, monkeypatch):
        """Non-positive threshold raises UserInputError."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "0")
        with pytest.raises(UserInputError):
            load_privacy_settings()

    def test_invalid_bin_width(self, monkeypatch):
        """Non-positive bin width raises UserInputError."""
        monkeypatch.setenv("COXPH_TIME_BIN_WIDTH", "0")
        with pytest.raises(UserInputError):
            load_privacy_settings()

    def test_invalid_k(self, monkeypatch):
        """k < 1 raises UserInputError."""
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "0")
        with pytest.raises(UserInputError):
            load_privacy_settings()

    def test_invalid_k_negative(self, monkeypatch):
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "-1")
        with pytest.raises(UserInputError):
            load_privacy_settings()

    def test_threshold_plus_one_less_than_k(self, monkeypatch):
        """T + 1 < k raises UserInputError."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "3")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "5")
        with pytest.raises(UserInputError, match="must be"):
            load_privacy_settings()

    def test_invalid_setting_error_not_chained(self, monkeypatch):
        """The settings error carries no chained cause (no traceback in the log)."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "0")
        with pytest.raises(UserInputError) as exc_info:
            load_privacy_settings()
        assert exc_info.value.__cause__ is None


@pytest.mark.unit
class TestLoadResourceCaps:
    """Tests for load_resource_caps."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("COXPH_MAX_COVARIATES", raising=False)
        monkeypatch.delenv("COXPH_MAX_GRID_POINTS", raising=False)

    def test_defaults(self, clean_env):
        caps = load_resource_caps()
        assert isinstance(caps, ResourceCaps)
        assert caps.max_covariates == 20
        assert caps.max_grid_points == 1000

    def test_custom_values(self, clean_env, monkeypatch):
        monkeypatch.setenv("COXPH_MAX_COVARIATES", "50")
        monkeypatch.setenv("COXPH_MAX_GRID_POINTS", "5000")
        caps = load_resource_caps()
        assert caps.max_covariates == 50
        assert caps.max_grid_points == 5000

    def test_invalid_values_raise(self, clean_env, monkeypatch):
        """Invalid cap values raise UserInputError naming the variable."""
        monkeypatch.setenv("COXPH_MAX_COVARIATES", "0")
        with pytest.raises(UserInputError, match="COXPH_MAX_COVARIATES"):
            load_resource_caps()
        monkeypatch.setenv("COXPH_MAX_COVARIATES", "many")
        with pytest.raises(UserInputError, match="COXPH_MAX_COVARIATES"):
            load_resource_caps()
        monkeypatch.delenv("COXPH_MAX_COVARIATES")
        monkeypatch.setenv("COXPH_MAX_GRID_POINTS", "-1")
        with pytest.raises(UserInputError, match="COXPH_MAX_GRID_POINTS"):
            load_resource_caps()


@pytest.mark.unit
class TestLoadCurveTimeSigFigs:
    """Tests for load_curve_time_sig_figs (central-side curve coarsening)."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("COXPH_CURVE_TIME_SIG_FIGS", raising=False)

    def test_default(self, clean_env):
        """The curve time keys are coarsened to 6 significant figures by default."""
        assert load_curve_time_sig_figs() == 6

    def test_custom_value(self, clean_env, monkeypatch):
        monkeypatch.setenv("COXPH_CURVE_TIME_SIG_FIGS", "10")
        assert load_curve_time_sig_figs() == 10

    def test_empty_string_reports_exact_times(self, clean_env, monkeypatch):
        """An empty COXPH_CURVE_TIME_SIG_FIGS reports the exact event times."""
        monkeypatch.setenv("COXPH_CURVE_TIME_SIG_FIGS", "")
        assert load_curve_time_sig_figs() is None

    def test_invalid_values_raise(self, clean_env, monkeypatch):
        """Invalid values raise UserInputError naming the variable."""
        monkeypatch.setenv("COXPH_CURVE_TIME_SIG_FIGS", "0")
        with pytest.raises(UserInputError, match="COXPH_CURVE_TIME_SIG_FIGS"):
            load_curve_time_sig_figs()
        monkeypatch.setenv("COXPH_CURVE_TIME_SIG_FIGS", "fine")
        with pytest.raises(UserInputError, match="COXPH_CURVE_TIME_SIG_FIGS"):
            load_curve_time_sig_figs()


@pytest.mark.unit
class TestLoadMaxHeldEventFraction:
    """Tests for load_max_held_event_fraction (the central abort cap)."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("COXPH_MAX_HELD_EVENT_FRACTION", raising=False)

    def test_default(self, clean_env):
        """The abort cap defaults to 0.5."""
        assert load_max_held_event_fraction() == 0.5

    def test_empty_string_disables_the_abort(self, clean_env, monkeypatch):
        """An empty COXPH_MAX_HELD_EVENT_FRACTION disables the abort."""
        monkeypatch.setenv("COXPH_MAX_HELD_EVENT_FRACTION", "")
        assert load_max_held_event_fraction() is None

    def test_custom_value(self, clean_env, monkeypatch):
        monkeypatch.setenv("COXPH_MAX_HELD_EVENT_FRACTION", "0.75")
        assert load_max_held_event_fraction() == 0.75
        monkeypatch.setenv("COXPH_MAX_HELD_EVENT_FRACTION", "1")
        assert load_max_held_event_fraction() == 1.0

    def test_invalid_values_raise(self, clean_env, monkeypatch):
        """Values outside (0, 1] raise UserInputError naming the variable."""
        for raw in ("0", "1.5", "-0.2", "abc", "nan", "inf"):
            monkeypatch.setenv("COXPH_MAX_HELD_EVENT_FRACTION", raw)
            with pytest.raises(UserInputError, match="COXPH_MAX_HELD_EVENT_FRACTION"):
                load_max_held_event_fraction()


@pytest.mark.unit
class TestCheckSampleSize:
    """Tests for check_sample_size."""

    @pytest.fixture
    def settings(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=None,
            min_risk_set_change=5,
            # These tests pin threshold semantics; the reconstruction row
            # bound has its own test class.
            reconstruction_row_factor=1,
        )

    def test_sufficient_rows_and_events(self, settings):
        df = pd.DataFrame({"event": [1] * 20 + [0] * 5})
        assert check_sample_size(df, "event", settings) is True

    def test_insufficient_rows(self, settings):
        df = pd.DataFrame({"event": [1] * 5})
        assert check_sample_size(df, "event", settings) is False

    def test_insufficient_events(self, settings):
        df = pd.DataFrame({"event": [1] * 5 + [0] * 50})
        assert check_sample_size(df, "event", settings) is False

    def test_rows_only_when_no_outcome(self, settings):
        """When outcome_col is None, only rows are checked."""
        df = pd.DataFrame({"x": range(20)})
        assert check_sample_size(df, None, settings) is True
        df_small = pd.DataFrame({"x": range(5)})
        assert check_sample_size(df_small, None, settings) is False

    def test_boundary_strictly_greater(self, settings):
        """Threshold is strict: exactly threshold events fails."""
        df = pd.DataFrame({"event": [1] * 10 + [0] * 50})
        # 10 events, threshold 10 -> 10 > 10 is False
        assert check_sample_size(df, "event", settings) is False

    def test_missing_outcome_col_raises(self, settings):
        """Missing outcome_col raises UserInputError."""
        df = pd.DataFrame({"x": range(20)})
        with pytest.raises(UserInputError, match="Outcome column"):
            check_sample_size(df, "event", settings)


@pytest.mark.unit
class TestReconstructionRowBound:
    """The sample-size check also enforces rows >= factor * k."""

    @pytest.fixture
    def settings(self):
        # Reconstruction bound = 10 * 5 = 50 rows.
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=None,
            min_risk_set_change=5,
            reconstruction_row_factor=10,
        )

    def test_boundary_inclusive(self, settings):
        """49 rows fails the bound, 50 rows meets it (events above threshold)."""
        df_49 = pd.DataFrame({"event": [1] * 49})
        assert check_sample_size(df_49, "event", settings) is False
        df_50 = pd.DataFrame({"event": [1] * 50})
        assert check_sample_size(df_50, "event", settings) is True

    def test_bound_uses_configured_factor_and_k(self):
        """The bound is factor * k of the loaded settings."""
        settings = PrivacySettings(
            sample_size_threshold=2,
            time_bin_width=None,
            min_risk_set_change=3,
            reconstruction_row_factor=4,
        )
        # Bound = 4 * 3 = 12 rows.
        assert check_sample_size(pd.DataFrame({"event": [1] * 11}), "event", settings) is False
        assert check_sample_size(pd.DataFrame({"event": [1] * 12}), "event", settings) is True

    def test_warning_names_bound_not_count(self, settings, capsys):
        """The warning names the configured bound, never the actual count."""
        # vantage6's warn writes to stdout rather than the logging module, so
        # the text is captured with capsys instead of caplog.
        df = pd.DataFrame({"event": [1] * 49})
        assert check_sample_size(df, "event", settings) is False
        out = capsys.readouterr().out
        assert "10 x k=5" in out
        assert "49" not in out


@pytest.mark.unit
class TestEnsureSpawnedByCentral:
    """Tests for the parent-task guard."""

    def _make_client(self, token=None, task_dict=None, raises=None):
        """Build a fake client with controllable token/task.get."""

        class FakeTask:
            def get(self, task_id):
                if raises is not None:
                    raise raises
                return task_dict

        class FakeClient:
            def __init__(self):
                self.task = FakeTask()
                if token is not None:
                    self._access_token = token

        return FakeClient()

    def _make_token(self, task_id):
        payload = {"sub": {"task_id": task_id}}
        return jwt.encode(payload, key="unused", algorithm="HS256")

    def test_mock_client_skipped(self):
        """A MockAlgorithmClient is skipped (in-process test)."""
        from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

        client = MockAlgorithmClient(
            datasets=[[{"database": pd.DataFrame({"x": [1]}), "db_type": "csv"}]],
            module="v6-cox-ph",
            organization_ids=[1],
        )
        # Should not raise
        ensure_spawned_by_central(client)

    def test_non_mock_client_without_token_fails_closed(self):
        """A non-mock client without _access_token raises AlgorithmError."""
        client = self._make_client(token=None)
        with pytest.raises(AlgorithmError, match="No access token"):
            ensure_spawned_by_central(client)

    def test_direct_invocation_rejected(self):
        """A user task (parent is None) raises PrivacyViolation."""
        token = self._make_token(task_id=7)
        client = self._make_client(token=token, task_dict={"parent": None})
        with pytest.raises(PrivacyViolation):
            ensure_spawned_by_central(client)

    def test_subtask_passes(self):
        """A task with a parent passes the guard."""
        token = self._make_token(task_id=7)
        client = self._make_client(token=token, task_dict={"parent": {"id": 3}})
        ensure_spawned_by_central(client)

    def test_task_lookup_failure_fails_closed(self):
        """If task.get raises, the guard fails closed with AlgorithmError."""
        token = self._make_token(task_id=7)
        client = self._make_client(token=token, raises=RuntimeError("connection error"))
        with pytest.raises(AlgorithmError):
            ensure_spawned_by_central(client)

    def test_bad_token_fails_closed(self):
        """An undecodable token fails closed with AlgorithmError."""
        client = self._make_client(token="not-a-jwt", task_dict={"parent": None})
        with pytest.raises(AlgorithmError):
            ensure_spawned_by_central(client)


@pytest.mark.unit
class TestValidateIterationInput:
    """Tests for validate_iteration_input."""

    @pytest.fixture
    def settings(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=None,
            min_risk_set_change=5,
        )

    @pytest.fixture
    def settings_binned(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=10.0,
            min_risk_set_change=5,
        )

    def test_valid_input(self, settings):
        beta = [0.1, -0.2]
        grid = [1.0, 2.0, 3.0]
        b, c, s, g = validate_iteration_input(beta, [0.0, 0.0], [1.0, 2.0], grid, ["a", "b"], settings)
        np.testing.assert_array_equal(b, [0.1, -0.2])
        np.testing.assert_array_equal(c, [0.0, 0.0])
        np.testing.assert_array_equal(s, [1.0, 2.0])
        assert g == [1.0, 2.0, 3.0]

    def test_beta_wrong_length(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1, 0.2], [0.0], [1.0, 1.0], [1.0], ["a", "b"], settings)

    def test_beta_non_finite(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1, np.nan], [0.0, 0.0], [1.0, 1.0], [1.0], ["a", "b"], settings)

    def test_unsorted_grid(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [0.0], [1.0], [3.0, 1.0], ["a"], settings)

    def test_duplicate_grid(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [0.0], [1.0], [1.0, 1.0], ["a"], settings)

    def test_nan_in_grid(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [0.0], [1.0], [1.0, float("nan")], ["a"], settings)

    def test_empty_grid(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [0.0], [1.0], [], ["a"], settings)

    def test_off_grid_with_binning(self, settings_binned):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [0.0], [1.0], [5.0], ["a"], settings_binned)

    def test_on_grid_with_binning(self, settings_binned):
        b, c, s, g = validate_iteration_input([0.1], [0.0], [1.0], [10.0, 20.0], ["a"], settings_binned)
        assert g == [10.0, 20.0]

    def test_centre_wrong_length(self, settings):
        with pytest.raises(UserInputError, match="centre must have length"):
            validate_iteration_input([0.1, 0.2], [0.0], [1.0, 1.0], [1.0], ["a", "b"], settings)

    def test_centre_non_finite(self, settings):
        with pytest.raises(UserInputError, match="centre contains non-finite"):
            validate_iteration_input([0.1, 0.2], [np.nan, 0.0], [1.0, 1.0], [1.0], ["a", "b"], settings)

    def test_scale_wrong_length(self, settings):
        with pytest.raises(UserInputError, match="scale must have length"):
            validate_iteration_input([0.1, 0.2], [0.0, 0.0], [1.0], [1.0], ["a", "b"], settings)

    def test_scale_non_positive(self, settings):
        with pytest.raises(UserInputError, match="scale must be strictly positive"):
            validate_iteration_input([0.1, 0.2], [0.0, 0.0], [1.0, 0.0], [1.0], ["a", "b"], settings)

    def test_scale_non_finite(self, settings):
        with pytest.raises(UserInputError, match="scale contains non-finite"):
            validate_iteration_input([0.1, 0.2], [0.0, 0.0], [1.0, np.inf], [1.0], ["a", "b"], settings)

    def test_error_not_chained_and_echoes_no_input(self, settings):
        """A rejected payload surfaces a clean error: no chained traceback, no input values."""
        with pytest.raises(UserInputError) as exc_info:
            validate_iteration_input([0.1], [0.0], [1.0], [7.5, 2.5], ["a"], settings)
        assert exc_info.value.__cause__ is None
        message = str(exc_info.value)
        assert "7.5" not in message
        assert "2.5" not in message


@pytest.mark.unit
class TestPrepareNodeData:
    """The shared node-preparation step runs every guard, in order.

    This is the executable form of the AGENTS.md rule that any step that
    removes rows runs before check_sample_size and identically in every
    partial: if the order or membership of this sequence changes, these
    tests fail.
    """

    @pytest.fixture
    def node_df(self):
        # 200 rows: above the default privacy level's admission bound
        # (max(10 x k=5, 3 x R=50) = 150 rows) so the threshold does not
        # trip in these guard-order tests. The sex column gives the
        # categorical guard-order test categories above the threshold.
        return pd.DataFrame(
            {
                "time": [float(i) for i in range(1, 201)],
                "event": [1] * 100 + [0] * 100,
                "age": [50.0] * 200,
                "treatment": [0, 1] * 100,
                "sex": ["M", "F"] * 100,
            }
        )

    @pytest.fixture
    def client(self):
        from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

        return MockAlgorithmClient(
            datasets=[[{"database": pd.DataFrame({"x": [1]}), "db_type": "csv"}]],
            module="v6-cox-ph",
            organization_ids=[1],
        )

    @pytest.fixture
    def default_env(self, monkeypatch):
        monkeypatch.delenv("SAMPLE_SIZE_THRESHOLD", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)

    def _record_guard_calls(self, monkeypatch):
        """Wrap the guards so their call order is recorded."""
        calls = []
        for name in [
            "ensure_spawned_by_central",
            "load_privacy_settings",
            "encode_covariates",
            "validate_expl_vars",
            "drop_incomplete_rows",
            "validate_survival_columns",
            "select_rows",
            "check_sample_size",
            "check_category_sizes",
        ]:
            original = getattr(privacy_guards, name)

            def make_wrapper(guard_name, guard_original):
                def wrapper(*args, **kwargs):
                    calls.append(guard_name)
                    return guard_original(*args, **kwargs)

                return wrapper

            monkeypatch.setattr(privacy_guards, name, make_wrapper(name, original))
        return calls

    def test_guard_order_with_expl_vars(self, client, node_df, default_env, monkeypatch):
        """All guards run, in the documented order."""
        calls = self._record_guard_calls(monkeypatch)
        covariates = resolve_expl_vars(["age", "treatment"])
        df, settings, threshold_met = prepare_node_data(client, node_df, "time", "event", covariates, need_outcome=True)
        assert calls == [
            "ensure_spawned_by_central",
            "load_privacy_settings",
            "encode_covariates",
            "validate_expl_vars",
            "drop_incomplete_rows",
            "validate_survival_columns",
            "select_rows",
            "check_sample_size",
            "check_category_sizes",
        ]
        assert threshold_met is True
        assert len(df) == 200

    def test_guard_order_with_categorical(self, client, node_df, default_env, monkeypatch):
        """A categorical covariate runs the same order; the category threshold passes."""
        # The minimal level keeps the admission bound at k rows: this test
        # asserts the guard ORDER, and the 100-row categories of node_df
        # cannot reach the standard level's bound of 150 on a 200-row node.
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "minimal")
        calls = self._record_guard_calls(monkeypatch)
        covariates = resolve_expl_vars(["age", {"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}])
        df, settings, threshold_met = prepare_node_data(client, node_df, "time", "event", covariates, need_outcome=True)
        assert calls == [
            "ensure_spawned_by_central",
            "load_privacy_settings",
            "encode_covariates",
            "validate_expl_vars",
            "drop_incomplete_rows",
            "validate_survival_columns",
            "select_rows",
            "check_sample_size",
            "check_category_sizes",
        ]
        assert threshold_met is True
        assert "sex[F]" in df.columns

    def test_guard_order_without_expl_vars(self, client, node_df, default_env, monkeypatch):
        """Without covariates, encode_covariates and validate_expl_vars are skipped."""
        calls = self._record_guard_calls(monkeypatch)
        _, _, threshold_met = prepare_node_data(client, node_df, "time", "event", None, need_outcome=True)
        assert calls == [
            "ensure_spawned_by_central",
            "load_privacy_settings",
            "drop_incomplete_rows",
            "validate_survival_columns",
            "select_rows",
            "check_sample_size",
            "check_category_sizes",
        ]
        assert threshold_met is True

    def test_hook_runs_before_threshold(self, client, node_df, default_env, monkeypatch):
        """Rows removed by the select_rows hook count against the threshold."""
        monkeypatch.setattr(
            privacy_guards,
            "select_rows",
            lambda df: df.iloc[:5],
        )
        covariates = resolve_expl_vars(["age", "treatment"])
        _, _, threshold_met = prepare_node_data(client, node_df, "time", "event", covariates, need_outcome=True)
        assert threshold_met is False

    def test_need_outcome_false_checks_rows_only(self, client, default_env, monkeypatch):
        """need_outcome=False checks rows only (perform_iteration today)."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "10")
        df = pd.DataFrame({"time": [1.0] * 200, "age": [50.0] * 200})
        covariates = resolve_expl_vars(["age"])
        _, _, threshold_met = prepare_node_data(client, df, "time", None, covariates, need_outcome=False)
        assert threshold_met is True
        df_small = pd.DataFrame({"time": [1.0] * 5, "age": [50.0] * 5})
        _, _, threshold_met = prepare_node_data(client, df_small, "time", None, covariates, need_outcome=False)
        assert threshold_met is False

    def test_nan_rows_dropped_before_threshold(self, client, default_env, monkeypatch):
        """Rows with NaN in any analysed column are dropped before the threshold."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "12")
        # The reconstruction row bound and the risk-set floor are switched
        # off: this test isolates the NaN-dropping / threshold semantics, not
        # the row bounds.
        monkeypatch.setenv("COXPH_RECONSTRUCTION_ROW_FACTOR", "1")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_SIZE", "")
        df = pd.DataFrame(
            {
                "time": [1.0] * 30,
                "event": [1] * 15 + [0] * 15,
                "age": [50.0] * 30,
                "treatment": [0, 1] * 15,
            }
        )
        # 12 rows get NaN in an analysed column: 15 events and 30 rows would
        # both pass threshold 12, but after dropping only 18 rows / 13 events
        # remain — rows still pass, events (13 > 12) pass too; push one more
        # NaN onto an event row to make events 12, which must fail (>12).
        df.loc[0:11, "age"] = np.nan
        df.loc[12, "age"] = np.nan  # row 12 is an event row
        covariates = resolve_expl_vars(["age", "treatment"])
        _, _, threshold_met = prepare_node_data(client, df, "time", "event", covariates, need_outcome=True)
        assert threshold_met is False


@pytest.mark.unit
class TestValidateSurvivalColumns:
    """Tests for validate_survival_columns (input rules, no value echoes)."""

    def _df(self, time=None, event=None):
        time = [1.0, 2.0, 3.0, 4.0] if time is None else time
        event = [1, 0, 1, 0] if event is None else event
        return pd.DataFrame({"time": time, "event": event})

    def test_valid_input_passes(self):
        validate_survival_columns(self._df(), "time", "event")

    def test_boolean_outcome_passes(self):
        validate_survival_columns(self._df(event=[True, False, True, False]), "time", "event")

    def test_missing_time_column_raises(self):
        with pytest.raises(UserInputError, match="Time column 'time' not found"):
            validate_survival_columns(pd.DataFrame({"event": [1]}), "time", "event")

    def test_missing_outcome_column_raises(self):
        with pytest.raises(UserInputError, match="Outcome column 'event' not found"):
            validate_survival_columns(pd.DataFrame({"time": [1.0]}), "time", "event")

    def test_no_outcome_col_skips_outcome_checks(self):
        df = pd.DataFrame({"time": [1.0, 2.0]})
        validate_survival_columns(df, "time", None)

    def test_string_time_raises(self):
        with pytest.raises(UserInputError, match="Time column 'time' must be numeric"):
            validate_survival_columns(self._df(time=["a", "b", "c", "d"]), "time", "event")

    def test_negative_time_raises(self):
        with pytest.raises(UserInputError, match="negative"):
            validate_survival_columns(self._df(time=[1.0, -2.0, 3.0, 4.0]), "time", "event")

    def test_infinite_time_raises(self):
        with pytest.raises(UserInputError, match="non-finite"):
            validate_survival_columns(self._df(time=[1.0, np.inf, 3.0, 4.0]), "time", "event")

    def test_outcome_coded_1_2_raises(self):
        with pytest.raises(UserInputError, match="only contain 0 and 1"):
            validate_survival_columns(self._df(event=[1, 2, 1, 0]), "time", "event")

    def test_string_outcome_raises(self):
        with pytest.raises(UserInputError, match="numeric or boolean"):
            validate_survival_columns(self._df(event=["yes", "no", "yes", "no"]), "time", "event")

    def test_messages_never_echo_values(self):
        """Error messages name columns and dtypes, never data values."""
        cases = [
            (self._df(time=[1.0, -123.456, 3.0, 4.0]), "time", "event", "-123.456"),
            (self._df(event=[1, 7, 1, 0]), "time", "event", "'7'"),
        ]
        for df, time_col, outcome_col, forbidden in cases:
            with pytest.raises(UserInputError) as exc_info:
                validate_survival_columns(df, time_col, outcome_col)
            assert forbidden not in str(exc_info.value)


@pytest.mark.unit
class TestBinTimes:
    """Tests for bin_times."""

    def test_no_width_returns_unchanged(self):
        times = pd.Series([1.5, 7.3, 12.8])
        result = bin_times(times, None)
        pd.testing.assert_series_equal(result, times)

    def test_zero_width_returns_unchanged(self):
        times = pd.Series([1.5, 7.3, 12.8])
        result = bin_times(times, 0)
        pd.testing.assert_series_equal(result, times)

    def test_bins_to_grid(self):
        times = pd.Series([1.5, 7.3, 12.8, 20.0])
        result = bin_times(times, 10.0)
        np.testing.assert_array_equal(result, [0.0, 0.0, 10.0, 20.0])

    def test_bins_negative(self):
        times = pd.Series([-3.2, 5.0])
        result = bin_times(times, 10.0)
        np.testing.assert_array_equal(result, [-10.0, 0.0])


@pytest.mark.unit
class TestTailCutoff:
    """Tests for tail_cutoff."""

    def test_disabled_when_k_le_one(self):
        times = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert tail_cutoff(times, k=1) is None
        assert tail_cutoff(times, k=0) is None

    def test_kth_largest(self):
        times = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # k=3 -> 3rd largest = 3.0
        assert tail_cutoff(times, k=3) == 3.0

    def test_kth_largest_with_duplicates(self):
        times = pd.Series([1.0, 2.0, 3.0, 3.0, 5.0])
        # sorted: [1,2,3,3,5], k=3 -> [-3] = 3.0
        assert tail_cutoff(times, k=3) == 3.0

    def test_fewer_than_k_raises(self):
        """Fewer than k valid times raises PrivacyViolation naming k, never the count."""
        times = pd.Series([1.0, 2.0])
        with pytest.raises(PrivacyViolation) as exc_info:
            tail_cutoff(times, k=5)
        message = str(exc_info.value)
        assert "5" in message
        assert "2" not in message


@pytest.mark.unit
class TestSuppressSmallTimeCounts:
    """Tests for suppress_small_time_counts (exact values)."""

    def test_min_count_one_is_noop(self):
        times = np.array([1.0, 2.0, 3.0])
        freqs = np.array([1, 5, 2])
        merged_times, merged_counts = suppress_small_time_counts(times, freqs, 1)
        np.testing.assert_array_equal(merged_times, times)
        np.testing.assert_array_equal(merged_counts, freqs)

    def test_merges_below_min_into_following_time(self):
        """Groups accumulate until they reach the minimum, at the largest time."""
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        freqs = np.array([1, 1, 3, 1, 1])
        merged_times, merged_counts = suppress_small_time_counts(times, freqs, 2)
        np.testing.assert_array_equal(merged_times, [2.0, 3.0, 5.0])
        np.testing.assert_array_equal(merged_counts, [2, 3, 2])
        assert merged_times.dtype == np.float64
        assert merged_counts.dtype == np.int64

    def test_trailing_small_group_merges_backward(self):
        """A trailing group below the minimum joins the previous emitted group."""
        times = np.array([1.0, 2.0])
        freqs = np.array([3, 1])
        merged_times, merged_counts = suppress_small_time_counts(times, freqs, 2)
        np.testing.assert_array_equal(merged_times, [1.0])
        np.testing.assert_array_equal(merged_counts, [4])

    def test_single_time_unchanged(self):
        merged_times, merged_counts = suppress_small_time_counts(np.array([7.0]), np.array([1]), 5)
        np.testing.assert_array_equal(merged_times, [7.0])
        np.testing.assert_array_equal(merged_counts, [1])

    def test_empty_input(self):
        merged_times, merged_counts = suppress_small_time_counts(np.array([]), np.array([]), 2)
        assert merged_times.shape == (0,)
        assert merged_counts.shape == (0,)
        assert merged_times.dtype == np.float64
        assert merged_counts.dtype == np.int64

    def test_counts_sum_preserved(self):
        """The event total is unchanged and the output grid stays ascending."""
        rng = np.random.default_rng(0)
        times = np.unique(np.sort(rng.uniform(0, 100, size=50)))
        freqs = rng.integers(1, 4, size=len(times))
        for min_count in (1, 2, 3, 5):
            merged_times, merged_counts = suppress_small_time_counts(times, freqs, min_count)
            assert merged_counts.sum() == freqs.sum()
            assert np.all(np.diff(merged_times) > 0)
            # Every emitted group reaches the minimum, except a grid that is a
            # single group in total.
            assert len(merged_counts) == 1 or np.all(merged_counts >= min_count)


@pytest.mark.unit
class TestMergedTimeMap:
    """Tests for merged_time_map (the suppression merge, exact values)."""

    def test_min_count_one_is_identity(self):
        times = np.array([1.0, 2.0, 3.0])
        mapped = merged_time_map(times, np.array([1, 5, 2]), 1)
        np.testing.assert_array_equal(mapped, times)
        assert mapped.dtype == np.float64

    def test_none_min_count_is_identity(self):
        times = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(merged_time_map(times, np.array([1, 5, 2]), None), times)

    def test_empty_input(self):
        mapped = merged_time_map(np.array([]), np.array([]), 2)
        assert mapped.shape == (0,)
        assert mapped.dtype == np.float64

    def test_groups_emit_at_largest_time(self):
        """Counts accumulate until they reach the minimum; the group maps to its largest time."""
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        freqs = np.array([1, 1, 3, 1, 1])
        np.testing.assert_array_equal(merged_time_map(times, freqs, 2), [2.0, 2.0, 3.0, 5.0, 5.0])

    def test_trailing_group_merges_backward(self):
        """A trailing group below the minimum maps to the previous emitted time."""
        times = np.array([1.0, 2.0])
        freqs = np.array([3, 1])
        np.testing.assert_array_equal(merged_time_map(times, freqs, 2), [1.0, 1.0])

    def test_whole_grid_one_group_at_last_time(self):
        """Nothing reaches the minimum: the whole grid merges to the last time."""
        times = np.array([1.0, 2.0, 3.0])
        freqs = np.array([1, 1, 1])
        np.testing.assert_array_equal(merged_time_map(times, freqs, 5), [3.0, 3.0, 3.0])

    def test_single_time_unchanged(self):
        np.testing.assert_array_equal(merged_time_map(np.array([7.0]), np.array([1]), 5), [7.0])

    def test_every_value_is_an_input_time(self):
        """The map is non-decreasing and every value is one of the input times."""
        rng = np.random.default_rng(0)
        times = np.unique(np.sort(rng.uniform(0, 100, size=50)))
        freqs = rng.integers(1, 4, size=len(times))
        for min_count in (1, 2, 3, 5):
            mapped = merged_time_map(times, freqs, min_count)
            assert mapped.shape == times.shape
            assert set(mapped.tolist()) <= set(times.tolist())
            assert np.all(np.diff(mapped) >= 0)


@pytest.mark.unit
class TestEventTimeCounts:
    """Tests for event_time_counts (the per-time event counts frame)."""

    def test_counts_per_time_sorted(self):
        df = pd.DataFrame({"time": [3.0, 1.0, 3.0, 2.0, 3.0, 1.0], "event": [1, 1, 0, 1, 1, 0], "x1": [0.0] * 6})
        counts = event_time_counts(df, "time", "event")
        assert counts.columns.tolist() == ["time", "freq"]
        assert counts["time"].tolist() == [1.0, 2.0, 3.0]
        assert counts["freq"].tolist() == [1, 1, 2]

    def test_only_event_rows_counted(self):
        df = pd.DataFrame({"time": [1.0, 1.0, 2.0], "event": [0, 0, 1]})
        counts = event_time_counts(df, "time", "event")
        assert counts["time"].tolist() == [2.0]
        assert counts["freq"].tolist() == [1]

    def test_no_events_empty_frame_with_columns(self):
        df = pd.DataFrame({"time": [1.0, 2.0], "event": [0, 0]})
        counts = event_time_counts(df, "time", "event")
        assert counts.columns.tolist() == ["time", "freq"]
        assert len(counts) == 0


@pytest.mark.unit
class TestPrepareTimeColumn:
    """Tests for prepare_time_column."""

    @pytest.fixture
    def settings_default(self):
        return PrivacySettings(sample_size_threshold=10, time_bin_width=None, min_risk_set_change=5)

    @pytest.fixture
    def settings_binned(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=10.0,
            min_risk_set_change=1,
        )

    @pytest.fixture
    def settings_k1(self):
        return PrivacySettings(sample_size_threshold=10, time_bin_width=None, min_risk_set_change=1)

    def test_no_binning_no_censor(self, settings_k1):
        df = pd.DataFrame({"time": [1.0, 5.0, 10.0], "event": [1, 0, 1]})
        out = prepare_time_column(df, "time", settings_k1, "event")
        pd.testing.assert_frame_equal(out, df)

    def test_does_not_modify_input(self, settings_binned):
        df = pd.DataFrame({"time": [1.5, 12.3], "event": [1, 1]})
        original = df.copy()
        prepare_time_column(df, "time", settings_binned, "event")
        pd.testing.assert_frame_equal(df, original)

    def test_bins_times(self, settings_binned):
        df = pd.DataFrame({"time": [1.5, 12.3], "event": [1, 0]})
        out = prepare_time_column(df, "time", settings_binned, "event")
        np.testing.assert_array_equal(out["time"].to_numpy(), [0.0, 10.0])

    def test_tail_censors_and_zeros_events(self, settings_default):
        # k=5, 10 rows; t_cut = 5th largest = 6.0; times 7-10 are clamped to 6.0
        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "event": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        out = prepare_time_column(df, "time", settings_default, "event")
        # Rows 6-9 (index) are clamped to 6.0 and their events zeroed
        for i in range(6, 10):
            assert out.loc[i, "time"] == 6.0
            assert out.loc[i, "event"] == 0
        # Row 5 (time 6.0) is unchanged
        assert out.loc[5, "time"] == 6.0
        assert out.loc[5, "event"] == 0

    def test_no_outcome_col_does_not_zero_events(self, settings_default):
        # k=5, 10 rows; t_cut = 6.0; times 7-10 clamped to 6.0
        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "other": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        out = prepare_time_column(df, "time", settings_default, outcome_col=None)
        # time clamped but no event column to zero
        for i in range(6, 10):
            assert out.loc[i, "time"] == 6.0
        assert "event" not in out.columns


@pytest.mark.unit
class TestPrepareTimeColumnMove:
    """Tests for the event-time move in prepare_time_column (move_event_times)."""

    @pytest.fixture
    def settings(self):
        # k=5 (tail censoring on), small-count suppression at the standard
        # level's minimum of 2.
        return PrivacySettings(sample_size_threshold=10, time_bin_width=None, min_risk_set_change=5)

    def test_events_move_to_merged_time(self, settings):
        """Event rows move up to their merged time; censored rows keep theirs."""
        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "event": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        out = prepare_time_column(df, "time", settings, "event", move_event_times=True)
        # Tail censoring (k=5): t_cut = 6.0, rows past it clamped and censored.
        # The three singleton events merge: the group reaches the minimum of 2
        # at time 2, and the trailing singleton at time 3 merges backward into
        # it, so all three events move to time 2.
        assert out["time"].tolist() == [2.0, 2.0, 2.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        assert out["event"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    def test_no_move_when_suppression_off(self, settings):
        """min_time_event_count <= 1 disables the move (tail censoring still runs)."""
        df = pd.DataFrame({"time": [float(t) for t in range(1, 11)], "event": [1, 1, 1] + [0] * 7})
        settings_off = PrivacySettings(
            sample_size_threshold=10, time_bin_width=None, min_risk_set_change=5, min_time_event_count=1
        )
        out = prepare_time_column(df, "time", settings_off, "event", move_event_times=True)
        assert out["time"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]

    def test_no_move_without_outcome_col(self, settings):
        """Without an outcome column there is nothing to move."""
        df = pd.DataFrame({"time": [float(t) for t in range(1, 11)], "other": [1] * 10})
        out = prepare_time_column(df, "time", settings, outcome_col=None, move_event_times=True)
        assert out["time"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]

    def test_default_is_no_move(self, settings):
        """move_event_times defaults to False (compute_summed_z's payload)."""
        df = pd.DataFrame({"time": [float(t) for t in range(1, 11)], "event": [1, 1, 1] + [0] * 7})
        out = prepare_time_column(df, "time", settings, "event")
        assert out["time"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]

    def test_merged_times_never_exceed_tail_cutoff(self, settings):
        """The tail censoring runs first: a merged time never exceeds t_cut."""
        # Events at 5 and 6 (count 1 each) merge to 6; t_cut = 5th largest = 6.
        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "event": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            }
        )
        out = prepare_time_column(df, "time", settings, "event", move_event_times=True)
        events = out[out["event"] == 1]
        assert events["time"].tolist() == [6.0, 6.0]
        assert events["time"].max() <= 6.0  # t_cut

    def test_integer_time_column_keeps_its_values(self, settings):
        """An integer time column keeps integer-valued times after the move."""
        df = pd.DataFrame({"time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "event": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]})
        out = prepare_time_column(df, "time", settings, "event", move_event_times=True)
        assert out["time"].tolist() == [2, 2, 2, 4, 5, 6, 6, 6, 6, 6]
        assert not out["time"].isna().any()

    def test_move_preserves_rows_and_events(self, settings):
        """The move changes times only: same rows, same events.

        All events sit at or below the tail cut-off, so the tail censoring
        leaves the outcome column alone and any change would come from the
        move itself.
        """
        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "event": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        out = prepare_time_column(df, "time", settings, "event", move_event_times=True)
        assert len(out) == len(df)
        assert out["event"].tolist() == df["event"].tolist()
        assert int(out["event"].sum()) == int(df["event"].sum())


@pytest.mark.unit
class TestGuardedRiskSetMasks:
    """Tests for guarded_risk_set_masks."""

    @pytest.fixture
    def times(self):
        # 10 individuals with distinct times
        return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def test_k1_is_plain_masks(self, times):
        grid = [2.0, 5.0, 8.0]
        masks = guarded_risk_set_masks(times, grid, k=1)
        assert len(masks) == 3
        assert masks[0].sum() == 9  # >= 2
        assert masks[1].sum() == 6  # >= 5
        assert masks[2].sum() == 3  # >= 8

    def test_holds_when_removed_below_k(self, times):
        """Moving from t to t+1 removing < k individuals holds the mask."""
        grid = [1.0, 2.0, 8.0]
        masks = guarded_risk_set_masks(times, grid, k=5)
        # At t=1: 10 at risk. At t=2: 9 at risk (removed 1 < 5) -> hold
        assert masks[0].sum() == 10
        assert masks[1].sum() == 10  # held
        # At t=8: 3 at risk (removed 6 from 10 >= 5) -> use cand
        assert masks[2].sum() == 3

    def test_consecutive_changes_are_zero_or_ge_k(self, times):
        """Every change in mask size is 0 or >= k."""
        grid = [float(i) for i in range(1, 11)]
        masks = guarded_risk_set_masks(times, grid, k=5)
        sizes = [int(m.sum()) for m in masks]
        for i in range(1, len(sizes)):
            change = sizes[i - 1] - sizes[i]
            assert change == 0 or change >= 5, f"change {change} at index {i} (sizes={sizes})"

    def test_tail_goes_to_empty_only_from_ge_k(self):
        """The final transition to an empty risk set must remove >= k."""
        times = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        grid = [1.0, 11.0]
        masks = guarded_risk_set_masks(times, grid, k=5)
        # t=1: 10 at risk. t=11: 0 at risk (removed 10 >= 5)
        assert masks[0].sum() == 10
        assert masks[1].sum() == 0


@pytest.mark.unit
class TestMinRiskSetFloor:
    """The absolute risk-set floor holds small risk sets (perform_iteration).

    At beta = 0 with a single covariate, agg1[t] equals the risk-set size
    |R(t)|, so the effective-index walk is observable directly on agg1.
    """

    @pytest.fixture
    def times(self):
        # 10 individuals with distinct times 1..10
        return pd.Series([float(i) for i in range(1, 11)])

    def _agg1(self, times, grid, k, min_risk_set_size):
        X = np.zeros((len(times), 1))
        beta = np.zeros(1)
        agg1, _, _ = guarded_risk_set_aggregates(times, grid, k, X, beta, min_risk_set_size)
        return agg1

    def test_holds_below_floor(self, times):
        """Risk sets below the floor are held at the last released aggregates."""
        grid = [float(i) for i in range(1, 11)]
        # |R| = [10, 9, ..., 1]; with k=5 the jump guard alone releases down
        # to 5, while the floor R=8 holds everything at the first set.
        agg1 = self._agg1(times, grid, k=5, min_risk_set_size=8)
        np.testing.assert_array_equal(agg1, np.full(10, 10.0))

    def test_jump_guard_alone_releases_smaller_sets(self, times):
        """Without the floor the same walk releases risk sets below R."""
        grid = [float(i) for i in range(1, 11)]
        agg1 = self._agg1(times, grid, k=5, min_risk_set_size=None)
        np.testing.assert_array_equal(agg1, [10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    def test_prefix_follows_jump_guard_until_floor_binds(self, times):
        """The released prefix follows the jump guard; the tail is held."""
        grid = [float(i) for i in range(1, 11)]
        # k=2 releases 10, 8, 6; below R=6 the walk holds at the last
        # released set (size 6).
        agg1 = self._agg1(times, grid, k=2, min_risk_set_size=6)
        np.testing.assert_array_equal(agg1, [10.0, 10.0, 8.0, 8.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])

    def test_first_grid_point_always_released(self):
        """The first grid point is released whatever its risk-set size."""
        times = pd.Series([1.0, 2.0, 3.0])
        grid = [1.0, 2.0, 3.0]
        # |R| = [3, 2, 1], all below the floor; only the first is released.
        agg1 = self._agg1(times, grid, k=5, min_risk_set_size=50)
        np.testing.assert_array_equal(agg1, [3.0, 3.0, 3.0])
        assert agg1[0] == 3.0  # below R=50, released anyway

    def test_k1_disables_the_floor(self, times):
        """k <= 1 disables the whole risk-set guard family, floor included."""
        grid = [float(i) for i in range(1, 11)]
        with_floor = self._agg1(times, grid, k=1, min_risk_set_size=8)
        without_floor = self._agg1(times, grid, k=1, min_risk_set_size=None)
        np.testing.assert_array_equal(with_floor, without_floor)
        # and they equal the plain risk-set sizes
        np.testing.assert_array_equal(with_floor, [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    def test_floor_none_matches_default_call(self, times):
        """min_risk_set_size=None is the default and leaves the walk unchanged."""
        grid = [float(i) for i in range(1, 11)]
        X = np.zeros((10, 1))
        beta = np.zeros(1)
        explicit = guarded_risk_set_aggregates(times, grid, 5, X, beta, None)[0]
        default = guarded_risk_set_aggregates(times, grid, 5, X, beta)[0]
        np.testing.assert_array_equal(explicit, default)

    def test_agg2_agg3_follow_the_same_effective_indices(self):
        """agg2/agg3 are indexed by the same effective grid indices as agg1."""
        times = pd.Series([1.0, 2.0, 3.0, 4.0])
        grid = [1.0, 2.0, 3.0, 4.0]
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        beta = np.zeros(1)
        # |R| = [4, 3, 2, 1]; k=2, R=2 -> effective = [0, 0, 2, 2]
        agg1, agg2, agg3 = guarded_risk_set_aggregates(times, grid, 2, X, beta, 2)
        np.testing.assert_array_equal(agg1, [4.0, 4.0, 2.0, 2.0])
        # S1 over the risk sets: {1,2,3,4} -> 10, {3,4} -> 7
        np.testing.assert_array_equal(agg2[:, 0], [10.0, 10.0, 7.0, 7.0])
        # S2 over the risk sets: 1+4+9+16=30, 9+16=25
        np.testing.assert_array_equal(agg3[:, 0, 0], [30.0, 30.0, 25.0, 25.0])


@pytest.mark.unit
class TestRestrictedGroupHolds:
    """The risk-set hold walk extends to category-restricted groups.

    The 0/1 indicator term of a categorical covariate lets a reader split
    the released sums into a category's members and the rest, so the walk
    holds a grid point while any group — or its complement — has fewer than
    k members leaving since the held point (its jump guard) or fewer than
    the floor members at risk (its floor). At beta = 0 with a single
    covariate, agg1[t] equals the risk-set size |R(t)|, so the walk is
    observable directly on agg1.
    """

    def _agg1(self, times, grid, k, min_risk_set_size, groups=None):
        X = np.zeros((len(times), 1))
        beta = np.zeros(1)
        agg1, _, _ = guarded_risk_set_aggregates(
            pd.Series(times), grid, k, X, beta, min_risk_set_size, restricted_groups=groups
        )
        return agg1

    def test_group_jump_guard_holds_small_member_differences(self):
        # 10 rows; the group holds the rows at times 1..5. Between grid 4
        # and 10 the total walk releases (6 rows leave >= k=5) but only 2
        # group members leave (< k) — the group jump guard holds.
        times = [float(i) for i in range(1, 11)]
        group = np.zeros(10, dtype=bool)
        group[:5] = True
        agg1 = self._agg1(times, [4.0, 10.0], k=5, min_risk_set_size=None, groups=[group])
        np.testing.assert_array_equal(agg1, [7.0, 7.0])  # held at the first set
        # without the groups the same walk releases the smaller set
        agg1_plain = self._agg1(times, [4.0, 10.0], k=5, min_risk_set_size=None)
        np.testing.assert_array_equal(agg1_plain, [7.0, 1.0])

    def test_group_floor_holds_small_member_risk_sets(self):
        # 12 rows; the group holds the rows at times 1..2. The total walk
        # releases 11, 8, 1 (k=2); the group holds everything: its jump
        # guard holds at grid 5 (1 member leaves < k=2) and its floor
        # holds from there on (no members at risk < R=3).
        times = [float(i) for i in range(1, 13)]
        group = np.zeros(12, dtype=bool)
        group[:2] = True
        agg1 = self._agg1(times, [2.0, 5.0, 12.0], k=2, min_risk_set_size=3, groups=[group])
        np.testing.assert_array_equal(agg1, [11.0, 11.0, 11.0])
        agg1_plain = self._agg1(times, [2.0, 5.0, 12.0], k=2, min_risk_set_size=3)
        np.testing.assert_array_equal(agg1_plain, [11.0, 8.0, 8.0])

    def test_complement_jump_guard_holds_single_nonmember_differences(self):
        # 8 rows; the group holds the rows at times 1..5. Between grid 4
        # and 7 the total walk releases (3 rows leave >= k=2) and 2 group
        # members leave (>= k), but exactly 1 non-member leaves — the
        # complement's jump guard holds.
        times = [float(i) for i in range(1, 9)]
        group = np.zeros(8, dtype=bool)
        group[:5] = True
        agg1 = self._agg1(times, [4.0, 7.0], k=2, min_risk_set_size=None, groups=[group])
        np.testing.assert_array_equal(agg1, [5.0, 5.0])
        agg1_plain = self._agg1(times, [4.0, 7.0], k=2, min_risk_set_size=None)
        np.testing.assert_array_equal(agg1_plain, [5.0, 2.0])

    def test_complement_floor_holds_small_nonmember_risk_sets(self):
        # 20 rows; the group holds the rows at times 1..8 and 11..18 (16
        # members), the rest at 9, 10, 19, 20. At grid 15.5 the total
        # risk set (5) clears the floor R=4, but only 2 non-members are at
        # risk (< R) — the complement's floor holds alone.
        times = [float(i) for i in range(1, 21)]
        group = np.zeros(20, dtype=bool)
        group[:8] = True
        group[10:18] = True
        agg1 = self._agg1(times, [10.5, 15.5], k=2, min_risk_set_size=4, groups=[group])
        np.testing.assert_array_equal(agg1, [10.0, 10.0])
        agg1_plain = self._agg1(times, [10.5, 15.5], k=2, min_risk_set_size=4)
        np.testing.assert_array_equal(agg1_plain, [10.0, 5.0])

    def test_no_groups_leaves_the_walk_unchanged(self):
        times = [float(i) for i in range(1, 11)]
        grid = [float(i) for i in range(1, 11)]
        explicit = self._agg1(times, grid, 5, 8, groups=None)
        default = self._agg1(times, grid, 5, 8)
        np.testing.assert_array_equal(explicit, default)

    def test_k1_disables_the_group_holds(self):
        times = [float(i) for i in range(1, 11)]
        group = np.zeros(10, dtype=bool)
        group[:5] = True
        with_groups = self._agg1(times, [4.0, 10.0], k=1, min_risk_set_size=8, groups=[group])
        without = self._agg1(times, [4.0, 10.0], k=1, min_risk_set_size=8)
        np.testing.assert_array_equal(with_groups, without)
        np.testing.assert_array_equal(with_groups, [7.0, 1.0])  # plain risk-set sizes

    def test_released_member_differences_are_zero_or_ge_k(self):
        """End-to-end on the guard: every released interval involves 0 or
        >= k members of the group, so no single member is isolated."""
        rng = np.random.default_rng(7)
        n = 60
        times = np.sort(rng.uniform(0.5, 50.0, size=n))
        group = np.zeros(n, dtype=bool)
        group[rng.choice(n, size=20, replace=False)] = True
        grid = sorted(set(np.round(times, 1)))
        k, floor = 5, 10
        agg1 = self._agg1(times, grid, k, floor, groups=[group])
        # released indices: where agg1 (the risk-set size at beta=0) changes
        released = [0] + [j for j in range(1, len(grid)) if agg1[j] != agg1[j - 1]]
        # member counts at risk per grid point
        member_at_risk = np.array([int((group & (times >= g)).sum()) for g in grid])
        for a, b in zip(released, released[1:]):
            leaving = int(member_at_risk[a] - member_at_risk[b])
            assert leaving == 0 or leaving >= k, f"interval ({grid[a]}, {grid[b]}] isolates {leaving} members"
        # and the guard actually engaged (the plain walk would isolate one)
        agg1_plain = self._agg1(times, grid, k, floor)
        released_plain = [0] + [j for j in range(1, len(grid)) if agg1_plain[j] != agg1_plain[j - 1]]
        isolated = [
            (a, b)
            for a, b in zip(released_plain, released_plain[1:])
            if 0 < int(member_at_risk[a] - member_at_risk[b]) < k
        ]
        assert isolated, "the test data must isolate a member without the group holds"


@pytest.mark.unit
class TestValidateExplVars:
    """Tests for validate_expl_vars (source columns first, then term columns)."""

    def test_valid_vars(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50], "tx": [1]})
        validate_expl_vars(df, resolve_expl_vars(["age", "tx"]), "time", "event")

    def test_missing_column(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50]})
        with pytest.raises(UserInputError, match="not found"):
            validate_expl_vars(df, resolve_expl_vars(["age", "missing"]), "time", "event")

    def test_missing_categorical_source_column_reports_its_own_name(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50]})
        covariates = resolve_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}])
        with pytest.raises(UserInputError, match="'sex' not found"):
            validate_expl_vars(df, covariates, "time", "event")

    def test_overlaps_time_col(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50]})
        with pytest.raises(UserInputError, match="must not equal time_col"):
            validate_expl_vars(df, resolve_expl_vars(["time"]), "time", "event")

    def test_overlaps_outcome_col(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50]})
        with pytest.raises(UserInputError, match="must not equal outcome_col"):
            validate_expl_vars(df, resolve_expl_vars(["event"]), "time", "event")

    def test_non_numeric_column(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "name": ["alice"]})
        with pytest.raises(UserInputError, match="must be numeric"):
            validate_expl_vars(df, resolve_expl_vars(["name"]), "time", "event")

    def test_non_numeric_term_column(self):
        """A categorical term column that is not numeric is refused."""
        df = pd.DataFrame({"time": [1.0], "event": [1], "sex": ["M"], "sex[F]": ["no"]})
        covariates = resolve_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}])
        with pytest.raises(UserInputError, match="'sex\\[F\\]' must be numeric"):
            validate_expl_vars(df, covariates, "time", "event")


@pytest.mark.unit
class TestResolveExplVars:
    """Tests for resolve_expl_vars (the node-side re-validation)."""

    def test_plain_names(self):
        resolved = resolve_expl_vars(["age", "treatment"])
        assert resolved.terms == ["age", "treatment"]

    def test_categorical_and_scored(self):
        resolved = resolve_expl_vars(
            [
                "age",
                {"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}},
                {"stage": {"scores": {"I": 1, "II": 2}}},
            ]
        )
        assert resolved.terms == ["age", "sex[F]", "stage"]

    def test_malformed_spec_raises_user_input_error(self):
        with pytest.raises(UserInputError, match="Invalid explanatory variables"):
            resolve_expl_vars([{"sex": {"categories": {"M": ["M"], "F": ["F"]}}}])  # no reference

    def test_not_a_list(self):
        with pytest.raises(UserInputError, match="Invalid explanatory variables"):
            resolve_expl_vars("age")


@pytest.mark.unit
class TestEncodeCovariates:
    """Tests for encode_covariates (the node-side encoding step)."""

    @pytest.fixture
    def sex_covariates(self):
        return resolve_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}])

    def test_categorical_encoding(self, sex_covariates):
        df = pd.DataFrame({"sex": ["M", "F", "M", "F"], "age": [1.0, 2.0, 3.0, 4.0]})
        out = encode_covariates(df, sex_covariates)
        assert out["sex"].tolist() == ["M", "F", "M", "F"]  # the source column is kept
        assert out["sex[F]"].tolist() == [0.0, 1.0, 0.0, 1.0]
        assert pd.api.types.is_float_dtype(out["sex[F]"])

    def test_numbers_given_as_numbers_or_text(self):
        covariates = resolve_expl_vars([{"grp": {"reference": "0", "categories": {"0": [0], "1": ["1"]}}}])
        df = pd.DataFrame({"grp": [0, 1, "1", "0"]})
        out = encode_covariates(df, covariates)
        assert out["grp[1]"].tolist() == [0.0, 1.0, 1.0, 0.0]

    def test_missing_codes_become_nan(self):
        covariates = resolve_expl_vars(
            [{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}, "missing": ["unknown"]}}]
        )
        df = pd.DataFrame({"sex": ["M", "unknown", "F"]})
        out = encode_covariates(df, covariates)
        assert out["sex[F]"].iloc[0] == 0.0
        assert pd.isna(out["sex[F]"].iloc[1])
        assert out["sex[F]"].iloc[2] == 1.0

    def test_unlisted_codes_become_nan(self, sex_covariates):
        # An unlisted code maps to NaN exactly like a listed missing code;
        # the row drops with the other incomplete rows.
        df = pd.DataFrame({"sex": ["M", "F", "X"]})
        out = encode_covariates(df, sex_covariates)
        assert pd.isna(out["sex[F]"].iloc[2])
        assert out["sex[F]"].iloc[0] == 0.0
        assert out["sex[F]"].iloc[1] == 1.0

    def test_scored_encoding(self):
        covariates = resolve_expl_vars([{"stage": {"scores": {"I": 1, "II": 2}, "missing": ["unknown"]}}])
        df = pd.DataFrame({"stage": ["I", "II", "unknown", "III"]})
        out = encode_covariates(df, covariates)
        assert out["stage"].iloc[0] == 1.0
        assert out["stage"].iloc[1] == 2.0
        assert pd.isna(out["stage"].iloc[2])
        assert pd.isna(out["stage"].iloc[3])

    def test_custom_key_is_used(self):
        """A custom key parameter is used (a test key that lowercases)."""

        def lower_key(value):
            key_value = code_key(value)
            return key_value.lower() if isinstance(key_value, str) else key_value

        covariates = resolve_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["m"], "F": ["f"]}}}])
        df = pd.DataFrame({"sex": ["M", "F", "m", "f"]})
        out = encode_covariates(df, covariates, key=lower_key)
        assert out["sex[F]"].tolist() == [0.0, 1.0, 0.0, 1.0]

    def test_colliding_codes_under_custom_key_raise(self):
        """Two codes that collide under a custom key are a spec error on the node."""

        def lower_key(value):
            key_value = code_key(value)
            return key_value.lower() if isinstance(key_value, str) else key_value

        # "M" and "m" do not collide under code_key, so the spec resolves;
        # under the lowercasing key they do.
        covariates = resolve_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["m"]}}}])
        df = pd.DataFrame({"sex": ["M", "F"]})
        with pytest.raises(UserInputError, match="are the same"):
            encode_covariates(df, covariates, key=lower_key)

    def test_missing_source_column_left_for_validate(self, sex_covariates):
        df = pd.DataFrame({"age": [1.0]})
        out = encode_covariates(df, sex_covariates)
        assert "sex[F]" not in out.columns
        with pytest.raises(UserInputError, match="not found"):
            validate_expl_vars(out, sex_covariates, "time", "event")

    def test_plain_names_untouched(self):
        """The encoder handles only categories and scores entries."""
        covariates = resolve_expl_vars(["age"])
        df = pd.DataFrame({"age": ["not", "numeric"]})
        out = encode_covariates(df, covariates)
        assert out["age"].tolist() == ["not", "numeric"]

    def test_input_not_modified(self, sex_covariates):
        df = pd.DataFrame({"sex": ["M", "F"]})
        original = df.copy()
        encode_covariates(df, sex_covariates)
        pd.testing.assert_frame_equal(df, original)

    def test_term_colliding_with_existing_column_raises(self):
        # A term that names an existing data column (here a bracketed time
        # column) is refused instead of silently overwriting it.
        covariates = resolve_expl_vars(
            [{"grp": {"reference": "base", "categories": {"base": ["base"], "rare": ["rare"]}}}]
        )
        df = pd.DataFrame({"grp": ["base", "rare"], "grp[rare]": [1.0, 2.0]})
        with pytest.raises(UserInputError, match="collides with an existing data column"):
            encode_covariates(df, covariates)

    def test_term_colliding_with_bystander_column_raises(self):
        covariates = resolve_expl_vars([{"x": {"reference": "0", "categories": {"0": ["0"], "1": ["1"]}}}])
        df = pd.DataFrame({"x": [0, 1], "x[1]": [5.0, 6.0]})
        with pytest.raises(UserInputError, match="collides with an existing data column"):
            encode_covariates(df, covariates)

    def test_no_collision_when_columns_are_new(self, sex_covariates):
        # The ordinary case: the term column does not exist yet.
        df = pd.DataFrame({"sex": ["M", "F"]})
        out = encode_covariates(df, sex_covariates)
        assert out["sex[F]"].tolist() == [0.0, 1.0]


@pytest.mark.unit
class TestCheckCategorySizes:
    """Tests for check_category_sizes (the per-category threshold)."""

    @pytest.fixture
    def settings(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=None,
            min_risk_set_change=5,
            reconstruction_row_factor=1,
        )

    @pytest.fixture
    def covariates(self):
        return resolve_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}])

    def _frame(self, sexes, events, covariates):
        df = pd.DataFrame(
            {
                "time": [float(i + 1) for i in range(len(sexes))],
                "event": events,
                "sex": sexes,
            }
        )
        return encode_covariates(df, covariates)

    def test_all_categories_large_enough(self, settings, covariates):
        # 15 M (12 events) and 15 F (12 events): both above threshold 10.
        df = self._frame(["M"] * 15 + ["F"] * 15, [1] * 12 + [0] * 3 + [1] * 12 + [0] * 3, covariates)
        check_category_sizes(df, "event", covariates, settings)

    def test_rows_at_threshold_fail(self, settings, covariates):
        # M has exactly 10 rows (threshold 10): not more -> fail.
        df = self._frame(["M"] * 10 + ["F"] * 20, [1] * 10 + [1] * 15 + [0] * 5, covariates)
        with pytest.raises(PrivacyThresholdViolation, match="'M'"):
            check_category_sizes(df, "event", covariates, settings)

    def test_events_at_threshold_fail(self, settings, covariates):
        # M has 15 rows but exactly 10 events -> fail (events side).
        df = self._frame(["M"] * 15 + ["F"] * 15, [1] * 10 + [0] * 5 + [1] * 12 + [0] * 3, covariates)
        with pytest.raises(PrivacyThresholdViolation, match="'M'"):
            check_category_sizes(df, "event", covariates, settings)

    def test_reference_category_checked(self, settings, covariates):
        # The reference (M) has 15 rows but 10 events -> fail.
        df = self._frame(["M"] * 15 + ["F"] * 15, [1] * 10 + [0] * 5 + [1] * 12 + [0] * 3, covariates)
        with pytest.raises(PrivacyThresholdViolation, match="'M'"):
            check_category_sizes(df, "event", covariates, settings)

    def test_non_reference_category_checked(self, settings, covariates):
        # F has 15 rows but 10 events -> fail.
        df = self._frame(["M"] * 15 + ["F"] * 15, [1] * 12 + [0] * 3 + [1] * 10 + [0] * 5, covariates)
        with pytest.raises(PrivacyThresholdViolation, match="'F'"):
            check_category_sizes(df, "event", covariates, settings)

    def test_empty_category_fails(self, settings):
        # A label no row holds: a zero is a disclosure as well. The other
        # categories pass, so the empty one is the failure.
        covariates = resolve_expl_vars(
            [{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"], "X": ["X"]}}}]
        )
        df = self._frame(["M"] * 11 + ["F"] * 11, [1] * 22, covariates)
        with pytest.raises(PrivacyThresholdViolation, match="'X'"):
            check_category_sizes(df, "event", covariates, settings)

    def test_message_names_threshold_variable_label_not_count(self, settings, covariates):
        # M has 7 rows (below threshold 10): the message names the threshold,
        # the variable and the label, never the count.
        df = self._frame(["M"] * 7 + ["F"] * 23, [1] * 7 + [1] * 16 + [0] * 7, covariates)
        with pytest.raises(PrivacyThresholdViolation) as exc_info:
            check_category_sizes(df, "event", covariates, settings)
        message = str(exc_info.value)
        assert "10" in message  # the configured threshold
        assert "sex" in message
        assert "'M'" in message
        assert "7" not in message  # never the count

    def test_no_categorical_covariates_is_a_noop(self, settings):
        covariates = resolve_expl_vars(["age"])
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50.0]})
        check_category_sizes(df, "event", covariates, settings)
        check_category_sizes(df, "event", None, settings)

    @pytest.fixture
    def hardened_settings(self):
        # Standard-level-like: factor 10, floor 50 -> admission bound 150.
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=None,
            min_risk_set_change=5,
            reconstruction_row_factor=10,
            min_risk_set_size=50,
        )

    def test_category_below_admission_bound_fails(self, hardened_settings, covariates):
        # 149 M rows (149 events) pass the threshold but not the bound 150;
        # the F category reaches the bound so M is the refused one.
        df = self._frame(["M"] * 149 + ["F"] * 151, [1] * 149 + [1] * 151, covariates)
        with pytest.raises(PrivacyThresholdViolation, match="'M'"):
            check_category_sizes(df, "event", covariates, hardened_settings)

    def test_category_at_admission_bound_passes(self, hardened_settings, covariates):
        # 150 M rows (150 events) reach the bound 150 and pass.
        df = self._frame(["M"] * 150 + ["F"] * 150, [1] * 150 + [1] * 150, covariates)
        check_category_sizes(df, "event", covariates, hardened_settings)

    def test_admission_bound_message_names_bounds_not_count(self, hardened_settings, covariates):
        # 149 M rows: the message names the threshold and the bound, never
        # the count.
        df = self._frame(["M"] * 149 + ["F"] * 151, [1] * 149 + [1] * 151, covariates)
        with pytest.raises(PrivacyThresholdViolation) as exc_info:
            check_category_sizes(df, "event", covariates, hardened_settings)
        message = str(exc_info.value)
        assert "10" in message  # the configured threshold
        assert "150" in message  # the configured admission bound
        assert "sex" in message
        assert "'M'" in message
        assert "149" not in message  # never the count


@pytest.mark.unit
class TestUnlistedCodesTreatedAsMissing:
    """Unlisted codes map to NaN and drop with the other incomplete rows.

    No error fires on data values: whether a node holds an unlisted code
    must not change the run's outcome (an error would be a per-value
    existence oracle). The rows drop identically in every partial, so the
    analysed row set stays consistent.
    """

    @pytest.fixture
    def client(self):
        from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

        return MockAlgorithmClient(
            datasets=[[{"database": pd.DataFrame({"x": [1]}), "db_type": "csv"}]],
            module="v6-cox-ph",
            organization_ids=[1],
        )

    @pytest.fixture
    def covariates(self):
        return resolve_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}])

    @pytest.fixture
    def env(self, monkeypatch):
        # Minimal level (no admission bound beyond factor * k) and threshold 10.
        monkeypatch.setenv("COXPH_PRIVACY_LEVEL", "minimal")
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "10")

    def test_unlisted_code_drops_its_row_without_error(self, client, covariates, env):
        # 22 M (11 events) and 22 F (11 events); one censored F row holds an
        # unlisted code. Every threshold passes; the row drops and the run
        # proceeds over the listed rows.
        sexes = ["M"] * 22 + ["F"] * 22
        events = [1] * 11 + [0] * 11 + [1] * 11 + [0] * 11
        df = pd.DataFrame({"time": [float(i + 1) for i in range(44)], "event": events, "sex": sexes})
        df.loc[43, "sex"] = "X"  # a censored F row
        df_out, _, threshold_met = prepare_node_data(client, df, "time", "event", covariates, need_outcome=True)
        assert threshold_met is True
        assert len(df_out) == 43  # the unlisted-code row dropped
        assert "sex[F]" in df_out.columns

    def test_unlisted_and_missing_codes_drop_identically(self, client, env):
        # A node holding an unlisted code and a node holding the same code
        # listed as missing analyse the same rows.
        covariates = resolve_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}])
        missing_covariates = resolve_expl_vars(
            [{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}, "missing": ["X"]}}]
        )
        sexes = ["M"] * 22 + ["F"] * 21 + ["X"]
        events = [1] * 11 + [0] * 11 + [1] * 11 + [0] * 10 + [0]
        df = pd.DataFrame({"time": [float(i + 1) for i in range(44)], "event": events, "sex": sexes})
        out_unlisted, _, met_unlisted = prepare_node_data(client, df, "time", "event", covariates, need_outcome=True)
        out_missing, _, met_missing = prepare_node_data(
            client, df, "time", "event", missing_covariates, need_outcome=True
        )
        assert met_unlisted is True and met_missing is True
        assert len(out_unlisted) == len(out_missing) == 43
        pd.testing.assert_frame_equal(out_unlisted, out_missing)

    def test_below_threshold_node_is_refused(self, client, covariates, env):
        """A node below the threshold is refused; no data-dependent error."""
        df = pd.DataFrame({"time": [1.0, 2.0, 3.0], "event": [1, 1, 0], "sex": ["M", "X", "F"]})
        _, _, threshold_met = prepare_node_data(client, df, "time", "event", covariates, need_outcome=True)
        assert threshold_met is False

    def test_category_below_threshold_still_fails(self, client, covariates, env):
        """The category threshold is unaffected by the unlisted code."""
        # 30 rows, 15 events; M has 5 rows (below threshold) and one row holds
        # an unlisted code.
        df = pd.DataFrame(
            {
                "time": [float(i + 1) for i in range(30)],
                "event": [1] * 15 + [0] * 15,
                "sex": ["M"] * 5 + ["F"] * 24 + ["X"],
            }
        )
        with pytest.raises(PrivacyThresholdViolation):
            prepare_node_data(client, df, "time", "event", covariates, need_outcome=True)

    def test_all_codes_listed_passes(self, client, covariates, env):
        sexes = ["M"] * 22 + ["F"] * 22
        events = [1] * 11 + [0] * 11 + [1] * 11 + [0] * 11
        df = pd.DataFrame({"time": [float(i + 1) for i in range(44)], "event": events, "sex": sexes})
        df_out, settings, threshold_met = prepare_node_data(client, df, "time", "event", covariates, need_outcome=True)
        assert threshold_met is True
        assert "sex[F]" in df_out.columns


@pytest.mark.unit
class TestDropIncompleteRows:
    """Tests for drop_incomplete_rows."""

    def test_drops_nan_rows(self):
        df = pd.DataFrame({"time": [1.0, np.nan, 3.0], "event": [1, 1, 0], "age": [50, 60, np.nan]})
        out = drop_incomplete_rows(df, ["time", "event", "age"])
        assert len(out) == 1
        assert out.index.tolist() == [0]

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"time": [1.0, np.nan], "event": [1, 1]})
        original = df.copy()
        drop_incomplete_rows(df, ["time", "event"])
        pd.testing.assert_frame_equal(df, original)

    def test_nonexistent_col_ignored(self):
        df = pd.DataFrame({"time": [1.0, 2.0], "event": [1, 0]})
        out = drop_incomplete_rows(df, ["time", "missing"])
        assert len(out) == 2


@pytest.mark.unit
class TestRiskSetEquivalence:
    """The vectorised aggregates equal the reference mask implementation."""

    def _random_case(self, seed: int, n_rows: int, n_grid: int, n_covs: int, k: int, tie: bool):
        """Build a random times/grid/covariates case with optional ties."""
        rng = np.random.default_rng(seed)
        grid = np.sort(rng.choice(np.arange(1, n_grid + 1) * 1.0, size=n_grid // 2, replace=False))
        times = rng.uniform(0.5, float(grid[-1]) + 1.0, size=n_rows)
        if tie:
            # Put some rows exactly on grid points and duplicate some times
            on_grid = rng.choice(len(grid), size=min(5, n_rows))
            times[on_grid] = grid[on_grid]
            times[: min(3, n_rows)] = times[0]
        X = rng.normal(size=(n_rows, n_covs))
        beta = rng.normal(size=n_covs)
        return pd.Series(times), grid.tolist(), X, beta

    @pytest.mark.parametrize("k", [1, 3, 5])
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_matches_reference(self, k, seed):
        """agg1/agg2/agg3 equal the mask implementation to rtol=1e-12."""
        from tests.unit.reference_risk_sets import mask_based_aggregates

        times, grid, X, beta = self._random_case(seed, n_rows=60, n_grid=30, n_covs=3, k=k, tie=True)

        ref1, ref2, ref3 = mask_based_aggregates(times, grid, k, X, beta)
        new1, new2, new3 = guarded_risk_set_aggregates(times, grid, k, X, beta)

        np.testing.assert_allclose(new1, ref1, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(new2, ref2, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(new3, ref3, rtol=1e-12, atol=1e-14)

    def test_rows_off_grid_and_before_grid(self):
        """Rows between grid points and before the first grid point count correctly."""
        from tests.unit.reference_risk_sets import mask_based_aggregates

        times = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 7.5])
        grid = [1.0, 2.0, 3.0, 5.0]
        X = np.arange(14, dtype=float).reshape(7, 2)
        beta = np.array([0.1, -0.2])

        for k in (1, 3):
            ref1, ref2, ref3 = mask_based_aggregates(times, grid, k, X, beta)
            new1, new2, new3 = guarded_risk_set_aggregates(times, grid, k, X, beta)
            np.testing.assert_allclose(new1, ref1, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(new2, ref2, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(new3, ref3, rtol=1e-12, atol=1e-14)

    def test_empty_risk_sets_match_reference(self):
        """Grid points beyond all rows give zero aggregates, as the masks did."""
        from tests.unit.reference_risk_sets import mask_based_aggregates

        times = pd.Series([1.0, 2.0, 3.0])
        grid = [1.0, 5.0, 9.0]
        X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        beta = np.zeros(2)

        ref1, ref2, ref3 = mask_based_aggregates(times, grid, 1, X, beta)
        new1, new2, new3 = guarded_risk_set_aggregates(times, grid, 1, X, beta)
        np.testing.assert_allclose(new1, ref1, rtol=1e-12, atol=1e-14)
        # The grid points beyond every row have empty risk sets
        assert new1[1] == 0.0 and new1[2] == 0.0


@pytest.mark.unit
class TestRoundReleasedAggregates:
    """Tests for round_released_aggregates (exact values)."""

    def test_none_returns_identical_arrays(self):
        agg1 = np.array([123.456, 0.0123456])
        agg2 = np.array([[1.23456, 2.34567], [3.45678, 4.56789]])
        agg3 = np.arange(8, dtype=float).reshape(2, 2, 2) / 7.0
        out1, out2, out3 = round_released_aggregates(agg1, agg2, agg3, None)
        assert out1 is agg1
        assert out2 is agg2
        assert out3 is agg3

    def test_sig_figs_rounds_exact_values(self):
        agg1 = np.array([123.456, 0.0123456, 0.0])
        agg2 = np.array([[1.0, -987.654]])
        agg3 = np.array([[[0.00123456, 2.0], [3.0, 4.0]]])
        out1, out2, out3 = round_released_aggregates(agg1, agg2, agg3, 3)
        np.testing.assert_array_equal(out1, [123.0, 0.0123, 0.0])
        np.testing.assert_array_equal(out2, [[1.0, -988.0]])
        np.testing.assert_array_equal(out3, [[[0.00123, 2.0], [3.0, 4.0]]])

    def test_shapes_and_dtypes_preserved(self):
        agg1 = np.array([123.456, 0.0123456])
        agg2 = np.array([[1.23456, 2.34567], [3.45678, 4.56789]])
        agg3 = np.arange(8, dtype=float).reshape(2, 2, 2) / 7.0
        out1, out2, out3 = round_released_aggregates(agg1, agg2, agg3, 4)
        assert out1.shape == agg1.shape
        assert out2.shape == agg2.shape
        assert out3.shape == agg3.shape
        assert out1.dtype == agg1.dtype
        assert out2.dtype == agg2.dtype
        assert out3.dtype == agg3.dtype


@pytest.mark.unit
class TestQuantiseWireAggregates:
    """Tests for quantise_wire_aggregates (exact values, always on)."""

    def test_quantises_to_float32_precision(self):
        agg1 = np.array([123.4567890123, 0.0123456789012, 0.0])
        agg2 = np.array([[1.0, -987.6543210987]])
        agg3 = np.array([[[0.00123456789012, 2.0], [3.0, 4.0]]])
        out1, out2, out3 = quantise_wire_aggregates(agg1, agg2, agg3)
        np.testing.assert_array_equal(out1, [123.4568, 0.01234568, 0.0])
        np.testing.assert_array_equal(out2, [[1.0, -987.6543]])
        np.testing.assert_array_equal(out3, [[[0.001234568, 2.0], [3.0, 4.0]]])

    def test_shapes_and_dtypes_preserved(self):
        agg1 = np.array([123.4567890123, 0.0123456789012])
        agg2 = np.array([[1.23456789012, 2.34567890123], [3.45678901234, 4.56789012345]])
        agg3 = np.arange(8, dtype=float).reshape(2, 2, 2) / 7.0
        out1, out2, out3 = quantise_wire_aggregates(agg1, agg2, agg3)
        assert out1.shape == agg1.shape
        assert out2.shape == agg2.shape
        assert out3.shape == agg3.shape
        assert out1.dtype == agg1.dtype
        assert out2.dtype == agg2.dtype
        assert out3.dtype == agg3.dtype

    def test_noop_on_coarser_values(self):
        """Values rounded to fewer figures (the standard/strict presets) pass through."""
        agg1 = np.array([123.456, 0.0123, 1.0, 0.0])
        agg2 = np.array([[1.0, -988.0]])
        agg3 = np.array([[[0.00123, 2.0], [3.0, 4.0]]])
        out1, out2, out3 = quantise_wire_aggregates(agg1, agg2, agg3)
        np.testing.assert_array_equal(out1, agg1)
        np.testing.assert_array_equal(out2, agg2)
        np.testing.assert_array_equal(out3, agg3)

    def test_idempotent(self):
        """Quantising an already quantised array changes nothing."""
        agg1 = np.array([123.4567890123, 0.0123456789012, 987.6543210987])
        agg2 = np.array([[1.23456789012, -2.34567890123]])
        agg3 = np.arange(8, dtype=float).reshape(2, 2, 2) / 7.0
        once1, once2, once3 = quantise_wire_aggregates(agg1, agg2, agg3)
        twice1, twice2, twice3 = quantise_wire_aggregates(once1, once2, once3)
        np.testing.assert_array_equal(twice1, once1)
        np.testing.assert_array_equal(twice2, once2)
        np.testing.assert_array_equal(twice3, once3)
