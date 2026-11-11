"""
Privacy guards for the federated Cox-PH partial functions.

These guards are pure functions (DataFrame/array in, DataFrame/array out)
so they can be unit-tested without vantage6. All partial functions run the
shared ``prepare_node_data`` step, in this order::

    ensure_spawned_by_central -> load_privacy_settings
    -> (encode_covariates, validate_expl_vars, where covariates are used)
    -> drop_incomplete_rows -> validate_survival_columns
    -> select_rows (hook) -> check_sample_size
    -> check_category_sizes (only when the threshold was met)
    -> (validate_iteration_input, perform_iteration only)
    -> prepare_time_column -> (function-specific work; perform_iteration
    builds its risk sets with guarded_risk_set_aggregates)

The settings are read from node environment variables (``algorithm_env``)
via ``get_env_var``:

- ``SAMPLE_SIZE_THRESHOLD`` (int, default 10) — a node is excluded / raises
  when its rows or events are not strictly greater than this threshold.
- ``COXPH_TIME_BIN_WIDTH`` (float, optional) — when set, event times are
  coarsened to a regular grid ``floor(t / w) * w``.
- ``COXPH_MIN_RISK_SET_CHANGE`` (int, default 5) — the minimum number of
  individuals by which consecutive shared risk-set aggregates must differ.
- ``COXPH_PRIVACY_LEVEL`` (string, default ``"standard"``, case-insensitive)
  — selects a preset bundle of hardening knobs. ``"minimal"`` leaves only the
  mandatory guards; ``"standard"`` is the default; ``"strict"`` hardens
  further. The presets are:

  - ``"minimal"``: ``reconstruction_row_factor`` 1 (off),
    ``min_time_event_count`` 1 (off), ``aggregate_sig_figs`` off (``None``),
    ``min_risk_set_size`` off (``None``).
  - ``"standard"``: ``reconstruction_row_factor`` 10,
    ``min_time_event_count`` 2, ``aggregate_sig_figs`` 6,
    ``min_risk_set_size`` 50 (the reconstruction bound ``factor * k``).
  - ``"strict"``: ``reconstruction_row_factor`` 20, ``min_time_event_count``
    5, ``aggregate_sig_figs`` 4, ``min_risk_set_size`` 100.

  Each knob can also be set individually; an explicitly set, non-empty
  variable beats the level preset:

- ``COXPH_RECONSTRUCTION_ROW_FACTOR`` (int >= 1) — the sample-size check also
  requires ``rows >= factor * k``, so a node stays many times larger than the
  smallest released group.
- ``COXPH_MIN_TIME_EVENT_COUNT`` (int >= 1) — per-time event counts below this
  are merged into the following time before they are shared. In
  ``perform_iteration`` the event rows' times are also moved up to their
  merged time (censored rows keep their time), so the risk sets match the
  shared grid; see ``prepare_time_column``.
- ``COXPH_AGGREGATE_SIG_FIGS`` (int >= 1; an empty string disables rounding) —
  released risk-set aggregates are rounded to this many significant figures.
- ``COXPH_MIN_RISK_SET_SIZE`` (int >= 1; an empty string disables the floor) —
  an absolute floor on the size of a released risk set: while fewer than this
  many individuals are at risk, the partial holds the last released
  aggregates instead of releasing a smaller set (the same hold pattern as the
  minimum-change guard; the first grid point is always released). The floor
  is part of the risk-set guard family: it only takes effect when
  ``COXPH_MIN_RISK_SET_CHANGE`` > 1, so ``k = 1`` disables it together with
  the jump guard. The sample-size check also requires ``rows >= 3 x R``
  while the floor is in effect (see ``check_sample_size``): a node that
  small would have (nearly) its whole content inside the smallest released
  risk set, which the floor alone cannot hide.

The released risk-set aggregates are additionally quantised to float32
precision (about 7 significant figures) at serialisation — always on, applied
after the privacy rounding above; see ``quantise_wire_aggregates``.

The resource caps are loaded centrally (``load_resource_caps``):

- ``COXPH_MAX_COVARIATES`` (int >= 1, default 20).
- ``COXPH_MAX_GRID_POINTS`` (int >= 1, default 1000).

The reported curve time keys are coarsened centrally as well
(``load_curve_time_sig_figs``):

- ``COXPH_CURVE_TIME_SIG_FIGS`` (int >= 1, default 6; an empty string
  reports the exact event times) — the time keys of the baseline
  cumulative hazard and the survival curves are rounded to this many
  significant figures; times that merge into one key keep the value at
  their largest original time.

The held-event fraction abort cap is loaded centrally too
(``load_max_held_event_fraction``):

- ``COXPH_MAX_HELD_EVENT_FRACTION`` (float in (0, 1], default 0.5; an
  empty string disables the abort) — above this share of a node's
  events scored against a held (unchanged) risk set, central aborts the
  run: the fit is dominated by held risk sets.
"""

from __future__ import annotations

from typing import Callable, Optional, TypedDict

import numpy as np
import pandas as pd
import jwt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from vantage6.algorithm.tools.exceptions import (
    AlgorithmError,
    PrivacyThresholdViolation,
    PrivacyViolation,
    UserInputError,
)

from .coxph_logic import round_sig
from .miscellaneous import ResolvedCovariates, build_code_map, code_key, format_validation_error
from vantage6.algorithm.tools.util import info, warn

# Environment variable names
ENV_SAMPLE_SIZE_THRESHOLD = "SAMPLE_SIZE_THRESHOLD"
ENV_TIME_BIN_WIDTH = "COXPH_TIME_BIN_WIDTH"
ENV_MIN_RISK_SET_CHANGE = "COXPH_MIN_RISK_SET_CHANGE"
ENV_MIN_RISK_SET_SIZE = "COXPH_MIN_RISK_SET_SIZE"
ENV_PRIVACY_LEVEL = "COXPH_PRIVACY_LEVEL"
ENV_RECONSTRUCTION_ROW_FACTOR = "COXPH_RECONSTRUCTION_ROW_FACTOR"
ENV_MIN_TIME_EVENT_COUNT = "COXPH_MIN_TIME_EVENT_COUNT"
ENV_AGGREGATE_SIG_FIGS = "COXPH_AGGREGATE_SIG_FIGS"
ENV_MAX_COVARIATES = "COXPH_MAX_COVARIATES"
ENV_MAX_GRID_POINTS = "COXPH_MAX_GRID_POINTS"
ENV_CURVE_TIME_SIG_FIGS = "COXPH_CURVE_TIME_SIG_FIGS"
ENV_MAX_HELD_EVENT_FRACTION = "COXPH_MAX_HELD_EVENT_FRACTION"

# Defaults
DEFAULT_SAMPLE_SIZE_THRESHOLD = 10
DEFAULT_MIN_RISK_SET_CHANGE = 5
DEFAULT_PRIVACY_LEVEL = "standard"
DEFAULT_MAX_COVARIATES = 20
DEFAULT_MAX_GRID_POINTS = 1000
DEFAULT_CURVE_TIME_SIG_FIGS = 6
DEFAULT_MAX_HELD_EVENT_FRACTION = 0.5

# The sample-size check admits a node only when it has at least this many
# times the risk-set size floor in rows (on top of the reconstruction bound
# factor * k): a smaller node would have (nearly) its whole content inside
# the smallest released risk set, which the floor alone cannot hide.
FLOOR_ADMISSION_MULTIPLE = 3

# The released aggregates travel to central as JSON numbers. Quantising them
# to float32 precision (about 7 significant decimal digits) at serialisation
# caps the payload size: a bare float32 cast would serialise to a longer
# float64 repr, so the cut comes from decimal quantisation. Always on,
# applied after the configurable privacy rounding (a coarser preset rounding
# makes it a no-op).
WIRE_AGGREGATE_SIG_FIGS = 7


# Hardening bundles selected by COXPH_PRIVACY_LEVEL. "minimal" leaves only the
# mandatory guards; "standard" is the default; "strict" hardens further.
class _LevelPreset(TypedDict):
    reconstruction_row_factor: int
    min_time_event_count: int
    aggregate_sig_figs: Optional[int]
    min_risk_set_size: Optional[int]


PRIVACY_LEVEL_PRESETS: dict[str, _LevelPreset] = {
    "minimal": {
        "reconstruction_row_factor": 1,
        "min_time_event_count": 1,
        "aggregate_sig_figs": None,
        "min_risk_set_size": None,
    },
    "standard": {
        "reconstruction_row_factor": 10,
        "min_time_event_count": 2,
        "aggregate_sig_figs": 6,
        "min_risk_set_size": 50,
    },
    "strict": {
        "reconstruction_row_factor": 20,
        "min_time_event_count": 5,
        "aggregate_sig_figs": 4,
        "min_risk_set_size": 100,
    },
}


class PrivacySettings(BaseModel):
    """Privacy-related settings read from the node environment.

    A frozen Pydantic model: the bounds (positive threshold, ``k >= 1``,
    positive bin width) and the threshold-covers-``k`` rule are declared on
    the fields instead of hand-rolled. The values still come in through
    vantage6's ``get_env_var`` (which decodes base32-encoded
    ``algorithm_env``); ``pydantic-settings`` would read ``os.environ``
    directly and bypass that decoding, so it is not used.

    ``privacy_level`` records which hardening bundle was selected. The
    four knobs the level prescribes — ``reconstruction_row_factor``,
    ``min_time_event_count``, ``aggregate_sig_figs`` and
    ``min_risk_set_size`` — are resolved in ``load_privacy_settings``: an
    explicitly set, non-empty environment variable beats the level
    preset, and an empty ``COXPH_AGGREGATE_SIG_FIGS`` or
    ``COXPH_MIN_RISK_SET_SIZE`` switches that knob off (``None``).
    """

    model_config = ConfigDict(frozen=True)

    sample_size_threshold: int = Field(gt=0)
    time_bin_width: Optional[float] = Field(default=None, gt=0)
    min_risk_set_change: int = Field(ge=1)
    privacy_level: str = Field(default=DEFAULT_PRIVACY_LEVEL)
    reconstruction_row_factor: int = Field(default=10, ge=1)
    min_time_event_count: int = Field(default=2, ge=1)
    aggregate_sig_figs: Optional[int] = Field(default=6, ge=1)
    min_risk_set_size: Optional[int] = Field(default=None, ge=1)

    @field_validator("privacy_level")
    @classmethod
    def _known_privacy_level(cls, v: str) -> str:
        if v not in PRIVACY_LEVEL_PRESETS:
            allowed = ", ".join(f"'{name}'" for name in PRIVACY_LEVEL_PRESETS)
            raise ValueError(f"privacy_level must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def _threshold_covers_k(self) -> "PrivacySettings":
        if self.sample_size_threshold + 1 < self.min_risk_set_change:
            raise ValueError(
                f"SAMPLE_SIZE_THRESHOLD ({self.sample_size_threshold}) + 1 must be "
                f">= COXPH_MIN_RISK_SET_CHANGE ({self.min_risk_set_change}); otherwise "
                f"a node can pass the sample-size threshold but fail to guarantee "
                f"risk sets of size >= k."
            )
        return self


class ResourceCaps(BaseModel):
    """Centrally enforced resource caps, read from the environment.

    Frozen Pydantic model; ``load_resource_caps`` supplies the defaults and
    validates the values (configuration, not data).
    """

    model_config = ConfigDict(frozen=True)

    max_covariates: int = Field(ge=1)
    max_grid_points: int = Field(ge=1)


def _parse_positive_int(raw, name: str) -> int:
    """Parse an ``int >= 1`` from an environment value, or raise UserInputError."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise UserInputError(f"{name} must be an integer >= 1, got {raw!r}") from None
    if value < 1:
        raise UserInputError(f"{name} must be an integer >= 1, got {raw!r}") from None
    return value


def _resolve_privacy_level(raw) -> str:
    """Normalise COXPH_PRIVACY_LEVEL (case-insensitive) or raise UserInputError."""
    level = str(raw).strip().lower()
    if level not in PRIVACY_LEVEL_PRESETS:
        allowed = ", ".join(f"'{name}'" for name in PRIVACY_LEVEL_PRESETS)
        raise UserInputError(f"{ENV_PRIVACY_LEVEL} must be one of {allowed} (case-insensitive), got {raw!r}")
    return level


def _resolve_knob(raw, preset_value: int, name: str) -> int:
    """Resolve a hardening knob: explicit env var (non-empty) beats the preset."""
    if raw is None or str(raw).strip() == "":
        return preset_value
    return _parse_positive_int(raw, name)


def _resolve_optional_knob(raw, preset_value: Optional[int], name: str) -> Optional[int]:
    """Resolve an optional hardening knob; an empty env var switches it off."""
    if raw is None:
        return preset_value
    if str(raw).strip() == "":
        return None
    return _parse_positive_int(raw, name)


def load_privacy_settings() -> PrivacySettings:
    """Load privacy settings from node environment variables.

    Uses ``get_env_var`` so that base32-encoded ``algorithm_env`` values are
    decoded the same way as in the rest of vantage6.

    The hardening knobs are resolved per knob: an explicitly set, non-empty
    environment variable beats the ``COXPH_PRIVACY_LEVEL`` preset, and an
    empty ``COXPH_AGGREGATE_SIG_FIGS`` or ``COXPH_MIN_RISK_SET_SIZE``
    disables that knob.

    Returns
    -------
    PrivacySettings
        Validated settings. ``time_bin_width`` is ``None`` when binning is
        disabled; ``aggregate_sig_figs`` and ``min_risk_set_size`` are
        ``None`` when the respective knob is disabled.

    Raises
    ------
    UserInputError
        If a setting value is not a positive number, the privacy level is
        not one of the allowed values, or a hardening knob is not an
        integer >= 1.
    """
    from vantage6.algorithm.tools.util import get_env_var

    sample_size_threshold = get_env_var(
        ENV_SAMPLE_SIZE_THRESHOLD,
        default=str(DEFAULT_SAMPLE_SIZE_THRESHOLD),
        as_type="int",
    )
    min_risk_set_change = get_env_var(
        ENV_MIN_RISK_SET_CHANGE,
        default=str(DEFAULT_MIN_RISK_SET_CHANGE),
        as_type="int",
    )
    time_bin_width_raw = get_env_var(ENV_TIME_BIN_WIDTH, default=None)
    time_bin_width: Optional[float] = None
    if time_bin_width_raw is not None and str(time_bin_width_raw).strip() != "":
        try:
            time_bin_width = float(time_bin_width_raw)
        except (TypeError, ValueError) as e:
            raise UserInputError(
                f"{ENV_TIME_BIN_WIDTH} must be a positive float, got " f"{time_bin_width_raw!r}"
            ) from e

    privacy_level = _resolve_privacy_level(get_env_var(ENV_PRIVACY_LEVEL, default=DEFAULT_PRIVACY_LEVEL))
    preset = PRIVACY_LEVEL_PRESETS[privacy_level]
    reconstruction_row_factor = _resolve_knob(
        get_env_var(ENV_RECONSTRUCTION_ROW_FACTOR, default=None),
        preset["reconstruction_row_factor"],
        ENV_RECONSTRUCTION_ROW_FACTOR,
    )
    min_time_event_count = _resolve_knob(
        get_env_var(ENV_MIN_TIME_EVENT_COUNT, default=None),
        preset["min_time_event_count"],
        ENV_MIN_TIME_EVENT_COUNT,
    )
    aggregate_sig_figs = _resolve_optional_knob(
        get_env_var(ENV_AGGREGATE_SIG_FIGS, default=None),
        preset["aggregate_sig_figs"],
        ENV_AGGREGATE_SIG_FIGS,
    )
    min_risk_set_size = _resolve_optional_knob(
        get_env_var(ENV_MIN_RISK_SET_SIZE, default=None),
        preset["min_risk_set_size"],
        ENV_MIN_RISK_SET_SIZE,
    )

    try:
        return PrivacySettings(
            sample_size_threshold=sample_size_threshold,
            time_bin_width=time_bin_width,
            min_risk_set_change=min_risk_set_change,
            privacy_level=privacy_level,
            reconstruction_row_factor=reconstruction_row_factor,
            min_time_event_count=min_time_event_count,
            aggregate_sig_figs=aggregate_sig_figs,
            min_risk_set_size=min_risk_set_size,
        )
    except ValidationError as e:
        # The message is formatted without the offending input; the chained
        # traceback would reach the container log the researcher reads, so
        # the cause is suppressed.
        raise UserInputError(f"Invalid privacy settings: {format_validation_error(e)}") from None


def load_resource_caps() -> ResourceCaps:
    """Load the resource caps from the environment (centrally enforced).

    Uses ``get_env_var`` like ``load_privacy_settings``. The caps bound the
    problem size the algorithm accepts; they are configuration, not data.

    Returns
    -------
    ResourceCaps
        Validated caps (``max_covariates``, ``max_grid_points``).

    Raises
    ------
    UserInputError
        If a cap is not an integer >= 1.
    """
    from vantage6.algorithm.tools.util import get_env_var

    max_covariates = _parse_positive_int(
        get_env_var(ENV_MAX_COVARIATES, default=str(DEFAULT_MAX_COVARIATES)),
        ENV_MAX_COVARIATES,
    )
    max_grid_points = _parse_positive_int(
        get_env_var(ENV_MAX_GRID_POINTS, default=str(DEFAULT_MAX_GRID_POINTS)),
        ENV_MAX_GRID_POINTS,
    )
    try:
        return ResourceCaps(max_covariates=max_covariates, max_grid_points=max_grid_points)
    except ValidationError as e:
        raise UserInputError(f"Invalid resource caps: {format_validation_error(e)}") from None


def load_curve_time_sig_figs() -> Optional[int]:
    """Load the curve time-key coarsening from the environment (centrally).

    Uses ``get_env_var`` like ``load_privacy_settings``. The baseline
    cumulative hazard and the survival curves are reported with their time
    keys rounded to this many significant figures; an empty
    ``COXPH_CURVE_TIME_SIG_FIGS`` reports the exact event times.

    Returns
    -------
    int | None
        The number of significant figures, or ``None`` for exact times.

    Raises
    ------
    UserInputError
        If the value is not an integer >= 1.
    """
    from vantage6.algorithm.tools.util import get_env_var

    return _resolve_optional_knob(
        get_env_var(ENV_CURVE_TIME_SIG_FIGS, default=None),
        DEFAULT_CURVE_TIME_SIG_FIGS,
        ENV_CURVE_TIME_SIG_FIGS,
    )


def load_max_held_event_fraction() -> Optional[float]:
    """Load the central abort cap on the held-event fraction.

    Uses ``get_env_var`` like ``load_privacy_settings``. Above this share
    of a node's events scored against a held (unchanged) risk set, central
    aborts the run with ``UserInputError``: the fit is dominated by held
    risk sets and the estimates are not trustworthy. An empty
    ``COXPH_MAX_HELD_EVENT_FRACTION`` disables the abort.

    Returns
    -------
    float | None
        The cap, or ``None`` when the abort is disabled.

    Raises
    ------
    UserInputError
        If the value is not a float in (0, 1].
    """
    from vantage6.algorithm.tools.util import get_env_var

    raw = get_env_var(ENV_MAX_HELD_EVENT_FRACTION, default=None)
    if raw is None:
        return DEFAULT_MAX_HELD_EVENT_FRACTION
    if str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise UserInputError(f"{ENV_MAX_HELD_EVENT_FRACTION} must be a float in (0, 1], got {raw!r}") from None
    if not 0.0 < value <= 1.0:
        raise UserInputError(f"{ENV_MAX_HELD_EVENT_FRACTION} must be a float in (0, 1], got {raw!r}")
    return value


def ensure_spawned_by_central(client) -> None:
    """Refuse to run a partial unless it was spawned by the ``central`` task.

    The container JWT ``sub`` claim carries the running ``task_id``. We look
    up that task and require a non-empty ``parent`` field: the vantage6
    server only sets ``parent_id`` for container-created sub-tasks, so a user
    task (called directly) has ``parent = None`` and is refused.

    The guard is skipped only for ``MockAlgorithmClient`` (in-process tests).
    Any other client without a decodable token raises ``AlgorithmError``
    (fail closed). Any failure to obtain the task also fails closed.

    Raises
    ------
    PrivacyViolation
        If the running task has no parent (i.e. it was created directly by a
        user rather than by ``central``).
    AlgorithmError
        If the token is missing or cannot be decoded, or if the task lookup
        itself fails (fail closed).
    """
    from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

    if isinstance(client, MockAlgorithmClient):
        info("Skipping parent-task guard: mock client detected.")
        return

    token = getattr(client, "_access_token", None)
    if token is None:
        raise AlgorithmError(
            "No access token found on client; cannot verify parent task. " "Refusing to proceed (fail closed)."
        )

    try:
        payload = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_sub": False,
            },
        )
        task_id = payload["sub"]["task_id"]
    except Exception as e:
        raise AlgorithmError(f"Could not decode task identity from container token: {e}") from e

    try:
        task = client.task.get(task_id)
    except Exception as e:
        raise AlgorithmError(f"Could not look up task {task_id} for parent-task guard: {e}") from e

    parent = task.get("parent") if isinstance(task, dict) else None
    if not parent:
        raise PrivacyViolation(
            "Partial functions may only be invoked as sub-tasks of the "
            "'central' function. Direct invocation is not permitted."
        )
    info("Parent-task guard passed: task has a parent.")


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Marked hook for project row selection.

    A project layer may replace this function to select the rows its
    protocol analyses (an eligibility filter, an inlier filter, …). The
    hook is a no-op here and runs **before** ``check_sample_size``, so any
    rows it removes count against the threshold. Implementations must
    never add rows and must remove rows only — otherwise the aggregates
    across partials describe different row sets.
    """
    return df


def resolve_expl_vars(expl_vars: list) -> ResolvedCovariates:
    """Re-validate the explanatory-variable spec on the node.

    The spec was validated at central entry; the node re-validates it (the
    wire is not trusted) and raises ``UserInputError`` on a malformed spec.
    The error messages echo only names, labels and codes of the request.

    Raises
    ------
    UserInputError
        On any malformed entry, duplicate name, clashing term or colliding
        code.
    """
    from .miscellaneous import parse_expl_vars

    try:
        return parse_expl_vars(expl_vars)
    except ValueError as e:
        raise UserInputError(f"Invalid explanatory variables: {e}") from None


def _map_codes(values: pd.Series, code_map: dict, key: Callable) -> pd.Series:
    """Map a source column through a code map.

    Returns the mapped values (the label or score of each listed code,
    ``None`` for a missing or unlisted code). Unlisted codes map to
    ``None`` exactly like listed ``missing`` codes, so the rows drop in
    the shared ``drop_incomplete_rows`` — no data-dependent error surface.
    """
    mapped: list = []
    for value in values:
        code_key_value = key(value)
        if code_key_value is None:
            mapped.append(None)
        elif code_key_value in code_map:
            mapped.append(code_map[code_key_value])
        else:
            mapped.append(None)
    return pd.Series(mapped, index=values.index, dtype=object)


def encode_covariates(
    df: pd.DataFrame,
    covariates: ResolvedCovariates,
    key: Callable = code_key,
) -> pd.DataFrame:
    """Encode the categorical and scored covariates into numeric term columns.

    A categorical variable maps each value to its category label and writes
    one float column ``"<name>[<label>]"`` per non-reference label (1.0/0.0);
    a scored variable maps each value to its score in the column named after
    the variable. Listed ``missing`` codes and unlisted codes become NaN in
    all of the variable's columns, so those rows drop with the other
    incomplete rows (identically in every partial); the source columns are
    kept. A model term that would overwrite an existing data column (the
    time or outcome column, or an unrelated column of the same name) is
    refused. A missing source column is left for ``validate_expl_vars``.
    Plain names are untouched: the encoder handles only ``categories`` and
    ``scores`` entries.

    Returns
    -------
    pd.DataFrame
        The frame with the term columns.

    Raises
    ------
    UserInputError
        When two listed codes collide under ``key`` (a spec error that
        depends on the request only, never on data), or when a model term
        would overwrite an existing data column (the time or outcome
        column, or an unrelated column whose name matches the term).
    """
    out = df.copy()
    existing_columns = set(out.columns)
    for entry in covariates.categorical:
        if entry.name not in out.columns:
            continue
        try:
            code_map = build_code_map(entry, key)
        except ValueError as e:
            raise UserInputError(str(e)) from None
        labels = _map_codes(out[entry.name], code_map, key)
        for label in entry.categories:
            if label == entry.reference:
                continue
            term = f"{entry.name}[{label}]"
            if term in existing_columns:
                raise UserInputError(
                    f"Model term '{term}' of explanatory variable '{entry.name}' collides with an "
                    f"existing data column; rename the category label or the column."
                )
            out[term] = (labels == label).astype(float)
            out.loc[labels.isna(), term] = np.nan
    for scored_entry in covariates.scored:
        if scored_entry.name not in out.columns:
            continue
        try:
            code_map = build_code_map(scored_entry, key)
        except ValueError as e:
            raise UserInputError(str(e)) from None
        scores = _map_codes(out[scored_entry.name], code_map, key)
        out[scored_entry.name] = scores.astype(float)
    return out


def check_category_sizes(
    df: pd.DataFrame,
    outcome_col: Optional[str],
    covariates: Optional[ResolvedCovariates],
    settings: PrivacySettings,
) -> None:
    """Check that every category of every categorical covariate is large enough.

    Every category — the reference included — must have strictly more rows
    and more events than ``settings.sample_size_threshold`` on the analysed
    rows, the same comparison ``check_sample_size`` uses. An empty category
    fails as well: a zero is a disclosure too.

    The rows must also reach the reconstruction admission bound
    (``_admission_bound``) — the same bound ``check_sample_size`` applies to
    whole nodes. A category's members are jointly isolated in the released
    risk-set sums (the 0/1 indicator term restricts S0/S1/S2 to them, and the
    first grid point is always released), so a category that small would be
    reconstructable member-by-member exactly as a small node is; the
    participation threshold alone does not bound that.

    The error names the variable, the category label and the configured
    bounds, never the count.

    Raises
    ------
    PrivacyThresholdViolation
        When a category does not exceed the threshold or reach the
        admission bound.
    """
    if covariates is None:
        return
    threshold = settings.sample_size_threshold
    admission_bound = _admission_bound(settings)
    for entry in covariates.categorical:
        if entry.name not in df.columns:
            continue  # validate_expl_vars already refused the missing column
        term_columns = [f"{entry.name}[{label}]" for label in entry.categories if label != entry.reference]
        if not all(term in df.columns for term in term_columns):
            continue
        masks = {label: df[f"{entry.name}[{label}]"] == 1.0 for label in entry.categories if label != entry.reference}
        masks[entry.reference] = (df[term_columns] == 0.0).all(axis=1)
        for label, mask in masks.items():
            n_rows = int(mask.sum())
            if n_rows <= threshold or n_rows < admission_bound:
                raise PrivacyThresholdViolation(
                    f"Category '{label}' of explanatory variable '{entry.name}' does not meet the "
                    f"configured sample-size threshold {threshold} and admission bound "
                    f"{admission_bound} (rows and events must both exceed the threshold; "
                    f"rows must also reach the admission bound)."
                )
            if outcome_col is not None:
                n_events = int((mask & (df[outcome_col] == 1)).sum())
                if n_events <= threshold:
                    raise PrivacyThresholdViolation(
                        f"Category '{label}' of explanatory variable '{entry.name}' does not exceed the "
                        f"configured sample-size threshold {threshold} (rows and events must both be greater)."
                    )


def prepare_node_data(
    client,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: Optional[str],
    covariates: Optional[ResolvedCovariates],
    *,
    need_outcome: bool,
) -> tuple[pd.DataFrame, PrivacySettings, bool]:
    """Run the shared node-side guard sequence once.

    Order (do not reorder or skip):

        ``ensure_spawned_by_central`` -> ``load_privacy_settings``
        -> ``encode_covariates`` (when covariates are given)
        -> ``validate_expl_vars`` (when covariates are given)
        -> ``drop_incomplete_rows`` -> ``validate_survival_columns``
        -> ``select_rows`` (hook) -> ``check_sample_size``
        -> ``check_category_sizes`` (only when the threshold was met)

    Every partial calls this so that all aggregates central combines come
    from the same row set on each node. The threshold failure behaviour
    (return a marker vs. raise) stays with the caller.

    Parameters
    ----------
    client : AlgorithmClient
        The algorithm client (used only by the parent-task guard).
    df : pd.DataFrame
        The node's data.
    time_col : str
        Name of the time column.
    outcome_col : str | None
        Name of the outcome column, or ``None`` when the partial does not
        receive it (``perform_iteration`` until the outcome column joins
        the wire contract).
    covariates : ResolvedCovariates | None
        The resolved explanatory variables (``resolve_expl_vars``), or
        ``None`` when the partial does not use covariates.
    need_outcome : bool
        Whether the sample-size threshold also counts events (rows and
        events must both exceed the threshold).

    Returns
    -------
    tuple[pd.DataFrame, PrivacySettings, bool]
        The prepared frame, the loaded privacy settings, and whether the
        sample-size threshold was met.
    """
    ensure_spawned_by_central(client)
    settings = load_privacy_settings()

    if covariates is not None:
        df = encode_covariates(df, covariates)
        validate_expl_vars(df, covariates, time_col, outcome_col)

    cols = [time_col]
    if outcome_col is not None:
        cols.append(outcome_col)
    if covariates is not None:
        cols.extend(covariates.terms)
    df = drop_incomplete_rows(df, cols)

    validate_survival_columns(df, time_col, outcome_col)

    df = select_rows(df)

    outcome = outcome_col if need_outcome else None
    threshold_met = check_sample_size(df, outcome, settings)

    if threshold_met:
        # The category threshold runs only when the sample-size threshold
        # was met: a node below it is refused for that reason alone.
        check_category_sizes(df, outcome, covariates, settings)

    return df, settings, threshold_met


def guarded_risk_set_aggregates(
    times: pd.Series,
    grid: list[float],
    k: int,
    X: np.ndarray,
    beta: np.ndarray,
    min_risk_set_size: Optional[int] = None,
    restricted_groups: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised risk-set aggregates with the minimum-change ("jump") guard.

    Rows are bucketed by grid point with ``searchsorted`` and summed per
    bucket with ``bincount`` (S0, S1 per covariate, S2 per covariate pair);
    reverse cumulative sums turn the bucket sums into risk-set sums. The
    jump guard becomes an *effective grid index* computed from the bucket
    counts alone: walking the grid while keeping the index ``h`` of the held
    risk set, ``removed = |R(h)| - |R(j+1)|``; if ``0 < removed < k`` the
    previous set is held (``h`` stays), otherwise ``h = j+1``. This matches
    the previous mask implementation exactly (see the reference copy in
    ``tests/unit/reference_risk_sets.py``).

    The absolute floor ``min_risk_set_size`` extends the same walk: while
    fewer than that many individuals are at risk at grid point ``j``, the
    last released aggregates are held instead of releasing a smaller risk
    set (no truncation — the tail is reported at the last released set,
    which makes the estimates approximate there). The floor belongs to the
    risk-set guard family: ``k <= 1`` disables it together with the jump
    guard, and the first grid point is always released whatever its
    risk-set size.

    ``restricted_groups`` extends the same walk to researcher-defined
    groups (the categories of the categorical covariates, reference
    included). The 0/1 indicator term of a category lets a reader separate
    the released sums into the group's members and the rest, so the walk
    holds a grid point while any group — or its complement — has fewer
    than ``k`` members leaving since the held point (its jump guard) or
    fewer than the floor members at risk (its floor). Released differences
    then involve 0 or at least ``k`` members of any category, and released
    group risk sets hold at least the floor — the same guarantee the total
    counts give, so category-restricted sums are no finer than the
    whole-node sums.

    Complexity: O(N·p²) compute and O(N·p + T·p²) memory, instead of T×N
    boolean masks.

    Parameters
    ----------
    times : pd.Series
        The node's (binned, tail-censored) times.
    grid : list[float]
        The sorted event-time grid.
    k : int
        The minimum risk-set change; ``k <= 1`` disables the guard.
    X : np.ndarray
        The (standardised) covariate matrix, shape (N, p).
    beta : np.ndarray
        The coefficient vector, shape (p,).
    min_risk_set_size : int | None
        The absolute floor on the size of a released risk set; ``None``
        disables the floor. Only effective when ``k > 1``.
    restricted_groups : list of array-like | None
        Boolean masks (length N), one per category of the categorical
        covariates (reference included): the rows of each researcher-
        defined group. ``None`` disables the group holds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``agg1`` (T,), ``agg2`` (T, p) and ``agg3`` (T, p, p): the shared
        risk-set sums S0(t) = sum exp(beta.x), S1(t) = sum x exp(beta.x) and
        S2(t) = sum x x^T exp(beta.x) per grid point.
    """
    grid_arr = np.asarray(grid, dtype=float)
    n_times = len(grid_arr)
    n_covs = X.shape[1]

    times_arr = pd.to_numeric(times, errors="coerce").to_numpy(dtype=float)
    weights = np.exp(X @ beta)

    # Bucket each row at the largest grid point that does not exceed its
    # time; a row in bucket b is in the risk sets of grid points 0..b.
    # Rows before the first grid point land in bucket -1 (in no risk set).
    finite = np.isfinite(times_arr)
    bucket = np.searchsorted(grid_arr, times_arr, side="right") - 1
    in_grid = (bucket >= 0) & finite
    bucket_clipped = np.where(in_grid, bucket, 0)

    # Per-bucket sums; reverse cumulative sums give the risk-set sums.
    counts = np.bincount(bucket_clipped[in_grid], minlength=n_times).astype(float)

    s0_bucket = np.bincount(bucket_clipped[in_grid], weights=weights[in_grid], minlength=n_times)
    s0 = np.cumsum(s0_bucket[::-1])[::-1]
    count_r = np.cumsum(counts[::-1])[::-1]  # |R(t_j)| per grid point

    s1_bucket = np.empty((n_times, n_covs))
    for c in range(n_covs):
        s1_bucket[:, c] = np.bincount(bucket_clipped[in_grid], weights=(weights * X[:, c])[in_grid], minlength=n_times)
    s1 = np.cumsum(s1_bucket[::-1], axis=0)[::-1]

    s2_bucket = np.empty((n_times, n_covs, n_covs))
    for c1 in range(n_covs):
        for c2 in range(c1, n_covs):
            s2_bucket[:, c1, c2] = np.bincount(
                bucket_clipped[in_grid], weights=(weights * X[:, c1] * X[:, c2])[in_grid], minlength=n_times
            )
            if c2 != c1:
                s2_bucket[:, c2, c1] = s2_bucket[:, c1, c2]
    s2 = np.cumsum(s2_bucket[::-1], axis=0)[::-1]

    # Per-group member counts at risk per grid point (reverse cumulative
    # sums of the per-bucket member counts), for the group holds below.
    group_count_r = []
    if restricted_groups:
        for mask in restricted_groups:
            member = np.asarray(mask, dtype=bool)
            sel = in_grid & member
            g_counts = np.bincount(bucket_clipped[sel], minlength=n_times).astype(float)
            group_count_r.append(np.cumsum(g_counts[::-1])[::-1])

    # Effective grid index from the counts alone (the jump guard, extended
    # by the absolute risk-set floor and by the category-restricted group
    # holds). The floor is part of the same guard family, so k <= 1
    # disables it together with the jump guard; the first grid point is
    # always released, whatever its risk-set size.
    floor = min_risk_set_size if (k is not None and k > 1) else None
    if k is None or k <= 1:
        effective = np.arange(n_times)
    else:
        effective = np.empty(n_times, dtype=int)
        effective[0] = 0
        held = 0
        for j in range(1, n_times):
            removed = count_r[held] - count_r[j]
            jump_holds = 0 < removed < k
            floor_holds = floor is not None and count_r[j] < floor
            group_holds = False
            for gcr in group_count_r:
                g_removed = gcr[held] - gcr[j]
                comp_removed = removed - g_removed
                if 0 < g_removed < k or 0 < comp_removed < k:
                    group_holds = True
                    break
                if floor is not None and (gcr[j] < floor or (count_r[j] - gcr[j]) < floor):
                    group_holds = True
                    break
            if not jump_holds and not floor_holds and not group_holds:
                held = j
            effective[j] = held

    agg1 = s0[effective]
    agg2 = s1[effective]
    agg3 = s2[effective]
    return agg1, agg2, agg3


def round_released_aggregates(agg1, agg2, agg3, sig_figs):
    """Round the released risk-set aggregates to ``sig_figs`` significant figures.

    The risk-set sums S0/S1/S2 leave the node at full float precision;
    rounding them to a configurable number of significant figures (from the
    privacy-level bundle, overridable through ``COXPH_AGGREGATE_SIG_FIGS``,
    an empty value disables the rounding) limits how precisely the
    aggregates pin down small groups. ``sig_figs=None`` returns the arrays
    unchanged. Shapes and dtypes are preserved.

    Parameters
    ----------
    agg1, agg2, agg3 : np.ndarray
        The released aggregates: ``agg1`` (T,), ``agg2`` (T, p) and ``agg3``
        (T, p, p).
    sig_figs : int | None
        Number of significant figures, or ``None`` to disable rounding.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The rounded ``agg1``, ``agg2`` and ``agg3``.
    """
    if sig_figs is None:
        return agg1, agg2, agg3
    return (
        round_sig(agg1, sig_figs),
        round_sig(agg2, sig_figs),
        round_sig(agg3, sig_figs),
    )


def quantise_wire_aggregates(agg1, agg2, agg3):
    """Quantise the released risk-set aggregates to float32 precision.

    The aggregates travel to central as JSON numbers; quantising them to
    about 7 significant decimal figures (float32 precision) at serialisation
    caps the payload size. The quantisation is always on and applies after
    the configurable privacy rounding (``round_released_aggregates``): where
    that rounding is coarser (the standard and strict presets) this is a
    no-op, and where it is off (the minimal preset) this is the only
    precision limit on the wire. Shapes and dtypes are preserved.

    Parameters
    ----------
    agg1, agg2, agg3 : np.ndarray
        The released aggregates: ``agg1`` (T,), ``agg2`` (T, p) and ``agg3``
        (T, p, p).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The quantised ``agg1``, ``agg2`` and ``agg3``.
    """
    return (
        round_sig(agg1, WIRE_AGGREGATE_SIG_FIGS),
        round_sig(agg2, WIRE_AGGREGATE_SIG_FIGS),
        round_sig(agg3, WIRE_AGGREGATE_SIG_FIGS),
    )


def validate_survival_columns(df: pd.DataFrame, time_col: str, outcome_col: Optional[str]) -> None:
    """Validate the survival columns on the prepared frame.

    The time column must be numeric, finite and non-negative; the outcome
    column must be binary (0/1 or boolean). A missing column raises
    ``UserInputError`` — it is a user-input problem that no fallback may
    hide. Error messages name columns and dtypes, never data values.

    Raises
    ------
    UserInputError
        On any validation failure.
    """
    if time_col not in df.columns:
        raise UserInputError(f"Time column '{time_col}' not found in data columns.")

    time_values = df[time_col]
    if not pd.api.types.is_numeric_dtype(time_values):
        raise UserInputError(f"Time column '{time_col}' must be numeric, got dtype {time_values.dtype}.")
    if not np.isfinite(time_values.to_numpy(dtype=float)).all():
        raise UserInputError(f"Time column '{time_col}' contains non-finite values.")
    if (time_values < 0).any():
        raise UserInputError(f"Time column '{time_col}' contains negative values.")

    if outcome_col is not None:
        if outcome_col not in df.columns:
            raise UserInputError(f"Outcome column '{outcome_col}' not found in data columns.")
        outcome_values = df[outcome_col]
        if not (pd.api.types.is_bool_dtype(outcome_values) or pd.api.types.is_numeric_dtype(outcome_values)):
            raise UserInputError(
                f"Outcome column '{outcome_col}' must be numeric or boolean, " f"got dtype {outcome_values.dtype}."
            )
        if not pd.api.types.is_bool_dtype(outcome_values) and not outcome_values.isin([0, 1]).all():
            raise UserInputError(
                f"Outcome column '{outcome_col}' must only contain 0 and 1 "
                f"(booleans are allowed); recode the data before calling."
            )


def _admission_bound(settings: PrivacySettings) -> int:
    """The reconstruction admission bound: the smallest group whose members
    the released aggregates may isolate whole.

    ``max(reconstruction_row_factor * k, FLOOR_ADMISSION_MULTIPLE * R)``
    with the floor term only while the risk-set floor is in effect
    (``COXPH_MIN_RISK_SET_CHANGE > 1``). Applied to whole nodes in
    ``check_sample_size`` and to every category in ``check_category_sizes``:
    a group that small would have (nearly) its whole content inside the
    smallest released risk set, and the floor would hold nearly its whole
    risk-set trajectory at the first grid point.
    """
    bound = settings.reconstruction_row_factor * settings.min_risk_set_change
    floor = settings.min_risk_set_size if settings.min_risk_set_change > 1 else None
    if floor is not None:
        bound = max(bound, FLOOR_ADMISSION_MULTIPLE * floor)
    return bound


def check_sample_size(df: pd.DataFrame, outcome_col: Optional[str], settings: PrivacySettings) -> bool:
    """Check whether the node meets the sample-size threshold.

    Both rows and events must be strictly greater than the threshold. The
    warning on failure names the threshold but never the actual count: the
    container log is returned to the researcher with the run, and a
    below-threshold count is exactly what the threshold protects.

    The sample-size threshold bounds *participation*, not reconstruction: a
    node whose entire content enters one released risk set can be
    reconstructed from the aggregates. A second, reconstruction bound
    therefore also applies — the node must have at least
    ``reconstruction_row_factor * k`` rows, which keeps the node many times
    larger than the smallest released group. When the risk-set size floor is
    in effect, the admission bound additionally requires
    ``FLOOR_ADMISSION_MULTIPLE * min_risk_set_size`` rows: a node that small
    would have (nearly) its whole content inside the smallest released risk
    set, and the floor would hold nearly the whole risk-set trajectory at
    the first grid point.

    Parameters
    ----------
    df : pd.DataFrame
        The node's data.
    outcome_col : str | None
        The outcome column name. When ``None`` (e.g. ``perform_iteration``
        where the outcome column is not available), only rows are checked.
    settings : PrivacySettings
        Loaded privacy settings.

    Returns
    -------
    bool
        ``True`` when rows > threshold, rows >= the admission bound
        (``max(factor * k, FLOOR_ADMISSION_MULTIPLE * R)`` with the floor
        term only while the floor is in effect) and (when checked) events >
        threshold.
    """
    threshold = settings.sample_size_threshold
    n_rows = len(df)
    if n_rows <= threshold:
        warn(f"Sample size threshold not met: row count does not exceed {threshold}.")
        return False

    # The admission bound covers both reconstruction routes; the warning
    # names the configured bound only, never the actual count.
    admission_bound = _admission_bound(settings)
    floor = settings.min_risk_set_size if settings.min_risk_set_change > 1 else None
    if n_rows < admission_bound:
        if floor is not None:
            warn(
                f"Sample size threshold not met: row count does not meet the admission bound "
                f"({settings.reconstruction_row_factor} x k={settings.min_risk_set_change}, "
                f"{FLOOR_ADMISSION_MULTIPLE} x R={floor}: at least {admission_bound} rows).",
            )
        else:
            warn(
                f"Sample size threshold not met: row count does not meet the reconstruction bound "
                f"({settings.reconstruction_row_factor} x k={settings.min_risk_set_change}).",
            )
        return False

    if outcome_col is not None:
        if outcome_col not in df.columns:
            raise UserInputError(f"Outcome column '{outcome_col}' not found in data columns.")
        n_events = int((df[outcome_col] == 1).sum())
        if n_events <= threshold:
            warn(f"Sample size threshold not met: event count does not exceed " f"{threshold}.")
            return False

    return True


def validate_iteration_input(
    beta,
    centre,
    scale,
    unique_time_events,
    expl_vars,
    settings: PrivacySettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Validate the wire input to ``perform_iteration``.

    A thin wrapper around the ``IterationInput`` Pydantic model: the model
    holds the wire shape (finite vectors, strictly positive scale, a
    strictly increasing grid, per-variable lengths and the bin-grid rule);
    this wrapper keeps the historical signature and return types
    (numpy arrays and a list of floats) and wraps ``ValidationError``
    into ``UserInputError``.

    Parameters
    ----------
    beta : array-like
        Current coefficient vector (standardised space).
    centre : array-like
        Pooled covariate means over the event cases.
    scale : array-like
        Pooled covariate standard deviations over the event cases.
    unique_time_events : list[float]
        The sorted event-time grid.
    expl_vars : list[str]
        Explanatory variable names (determine the expected lengths).
    settings : PrivacySettings
        Loaded privacy settings (used for the bin-grid check).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]
        The validated ``beta``, ``centre`` and ``scale`` as numpy arrays and
        the validated ``unique_time_events`` as a list of floats.

    Raises
    ------
    UserInputError
        On any validation failure reported by ``IterationInput``.
    """
    from .miscellaneous import IterationInput

    try:
        IterationInput.model_validate(
            {"beta": beta, "centre": centre, "scale": scale, "unique_time_events": unique_time_events},
            context={"expl_vars": expl_vars, "settings": settings},
        )
    except ValidationError as e:
        # Formatted without the offending input, and without the chained
        # traceback that would reach the container log.
        raise UserInputError(format_validation_error(e)) from None

    beta_arr = np.asarray(beta, dtype=float)
    centre_arr = np.asarray(centre, dtype=float)
    scale_arr = np.asarray(scale, dtype=float)
    return beta_arr, centre_arr, scale_arr, [float(t) for t in unique_time_events]


def bin_times(times: pd.Series, width: float | None) -> pd.Series:
    """Coarsen event times to a regular grid ``floor(t / w) * w``.

    When ``width`` is ``None`` or non-positive the times are returned
    unchanged.
    """
    if width is None or width <= 0:
        return times
    return pd.Series(
        np.floor(times.to_numpy(dtype=float) / width) * width,
        index=times.index,
    )


def tail_cutoff(times: pd.Series, k: int) -> float | None:
    """Return the k-th largest time, used as the tail-censoring cut-off.

    Rows with ``time > t_cut`` are administratively censored (their time is
    clamped to ``t_cut``), which guarantees that the largest shared risk set
    contains at least ``k`` individuals.

    Returns ``None`` when ``k <= 1`` (guard disabled).

    Raises ``PrivacyViolation`` when ``k > 1`` and fewer than ``k`` valid
    times remain — a node should have been excluded by the sample-size
    threshold before this is reached, so this is a defence-in-depth
    assertion.
    """
    if k is None or k <= 1:
        return None
    vals = pd.to_numeric(times, errors="coerce").dropna().to_numpy()
    if len(vals) < k:
        # The message names the configured k only; the actual number of valid
        # times is a count and must not reach the container log.
        raise PrivacyViolation(f"Tail censoring requires at least {k} valid times; fewer are available.")
    return float(np.sort(vals)[-k])


def merged_time_map(times, freqs, min_count: int) -> np.ndarray:
    """Map each event time to its merged time under small-cell suppression.

    The single source of truth for the suppression merge, used both by
    ``suppress_small_time_counts`` (which reduces this map to the unique
    merged times and their summed counts) and by the event-time move in
    ``prepare_time_column``: walking the ascending grid, a group accumulates
    counts until it reaches ``min_count`` and is then emitted at its largest
    time — every time of the group maps to that emitted time. A trailing
    group that never reaches ``min_count`` merges backward into the previous
    emitted group (its times map to that group's time); when nothing was
    emitted, the whole grid is one group at the last time.

    Parameters
    ----------
    times : array-like
        Sorted unique event times (float, ascending).
    freqs : array-like
        Positive integer event counts, one per time.
    min_count : int
        The minimum released per-time count. ``min_count <= 1`` (or ``None``)
        maps every time to itself.

    Returns
    -------
    np.ndarray
        The merged time per input time, float64, same length as ``times``;
        every value is one of the input times.
    """
    times_arr = np.asarray(times, dtype=float)
    mapped = times_arr.astype(np.float64)
    if min_count is None or min_count <= 1 or times_arr.size == 0:
        return mapped

    group_count = 0
    group_start = 0
    emitted_time: Optional[float] = None
    for i in range(times_arr.size):
        group_count += int(freqs[i])
        if group_count >= min_count:
            # Times ascend, so times_arr[i] is the largest time of the group.
            mapped[group_start : i + 1] = times_arr[i]
            emitted_time = float(times_arr[i])
            group_count = 0
            group_start = i + 1
    if group_count > 0:
        # Trailing group below the minimum: merge it backward into the
        # previous emitted group; when nothing was emitted the whole grid is
        # one group at the last time.
        mapped[group_start:] = emitted_time if emitted_time is not None else times_arr[-1]
    return mapped


def suppress_small_time_counts(times, freqs, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Suppress small per-time event counts on the event-time grid.

    Small-cell suppression on the shared event-time grid: a per-time event
    count of 1 identifies a singleton. Times whose count is below
    ``min_count`` are merged into the *following* time — a group's total
    count is accumulated until it reaches ``min_count``, and the group is
    then emitted at its largest time with the summed count. Merged cells tie
    those events at the group's largest time (the same kind of small
    perturbation as time binning) and the event total is preserved, so
    ``n_events`` is unchanged. If the trailing group never reaches
    ``min_count`` and at least one group was already emitted, it is merged
    backward into the previous emitted group (its count is added, the
    previous group's time is kept); a single-time grid is emitted unchanged.
    The merge itself is defined by ``merged_time_map``; this function reduces
    its output to the unique merged times with their summed counts.

    Parameters
    ----------
    times : array-like
        Sorted unique event times (float, ascending).
    freqs : array-like
        Positive integer event counts, one per time.
    min_count : int
        The minimum released per-time count. ``min_count <= 1`` returns the
        inputs unchanged.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(merged_times, merged_counts)`` as float64 / int64 arrays,
        ascending, with the counts summing to the input total.
    """
    times_arr = np.asarray(times, dtype=float)
    freqs_arr = np.asarray(freqs)
    if min_count is None or min_count <= 1 or times_arr.size == 0:
        return times_arr.astype(np.float64), freqs_arr.astype(np.int64)

    mapped = merged_time_map(times_arr, freqs_arr, min_count)
    merged_times = np.unique(mapped)
    merged_counts = np.zeros(merged_times.shape, dtype=np.int64)
    np.add.at(merged_counts, np.searchsorted(merged_times, mapped), freqs_arr.astype(np.int64))
    return merged_times, merged_counts


def event_time_counts(df: pd.DataFrame, time_col: str, outcome_col: str) -> pd.DataFrame:
    """Per-time event counts of a node's (prepared) DataFrame.

    The event rows (``outcome == 1``) grouped by the time column and sorted
    by time, as a DataFrame with columns ``[time_col, "freq"]`` — the same
    shape as the ``times`` payload of ``compute_summed_z``. Both partials
    derive the event grid from the same prepared frame, so the shared grid
    and the risk sets describe the same rows.

    Parameters
    ----------
    df : pd.DataFrame
        The node's prepared data (binned and tail-censored times).
    time_col : str
        Name of the time column.
    outcome_col : str
        Name of the outcome column.

    Returns
    -------
    pd.DataFrame
        Columns ``[time_col, "freq"]``, ascending by time; empty (with the
        same columns) when the frame holds no events.
    """
    events = df[df[outcome_col] == 1]
    counts = events.groupby(time_col, as_index=False).count()
    counts = counts.sort_values(by=time_col)[[time_col, outcome_col]]
    counts["freq"] = counts[outcome_col]
    return counts.drop(columns=outcome_col)


def prepare_time_column(
    df: pd.DataFrame,
    time_col: str,
    settings: PrivacySettings,
    outcome_col: str | None = None,
    *,
    move_event_times: bool = False,
) -> pd.DataFrame:
    """Bin and tail-censor the time column of a node's DataFrame.

    The transformation is applied in this order: bin the time column to the
    configured grid, compute the tail cut-off from the (binned) times, clamp
    rows past the cut-off to ``t_cut`` and, when an outcome column is given,
    set their event to 0 (administrative censoring).

    With ``move_event_times=True`` and an outcome column, the small-count
    suppression is made consistent: the node's merged event times are
    recomputed deterministically from its event counts (``merged_time_map``)
    and the event rows' times are moved up to their merged time, while
    censored rows keep their time. The analysed rows, the shared event grid
    and the risk sets then describe the same event times, so the partial
    likelihood matches the suppressed ``times`` payload shared by
    ``compute_summed_z``. The move applies whenever ``min_time_event_count >
    1``, whatever the risk-set guards; the tail censoring runs first, so a
    merged time never exceeds ``t_cut``.

    The input is not modified; a copy is returned.
    """
    out = df.copy()

    width = settings.time_bin_width
    if width is not None and width > 0:
        out[time_col] = bin_times(out[time_col], width)

    t_cut = tail_cutoff(out[time_col], settings.min_risk_set_change)
    if t_cut is not None:
        clamp_mask = out[time_col] > t_cut
        if clamp_mask.any():
            out.loc[clamp_mask, time_col] = t_cut
            if outcome_col is not None and outcome_col in out.columns:
                out.loc[clamp_mask, outcome_col] = 0

    if move_event_times and outcome_col is not None and settings.min_time_event_count > 1:
        counts = event_time_counts(out, time_col, outcome_col)
        if not counts.empty:
            time_values = counts[time_col].to_numpy(dtype=float)
            mapped = merged_time_map(time_values, counts["freq"].to_numpy(), settings.min_time_event_count)
            move_to = dict(zip(time_values.tolist(), mapped.tolist()))
            event_mask = out[outcome_col] == 1
            out.loc[event_mask, time_col] = out.loc[event_mask, time_col].map(move_to)

    return out


def validate_expl_vars(
    df: pd.DataFrame,
    covariates: ResolvedCovariates,
    time_col: str,
    outcome_col: Optional[str],
) -> None:
    """Validate the explanatory variables on the node.

    First checks that each source column exists (reporting it by its own
    name) and does not overlap ``time_col`` or ``outcome_col``; then that
    each model-term column is numeric. Column names are user input and may
    be echoed in the error message; data values are never echoed.

    Raises
    ------
    UserInputError
        On any validation failure.
    """
    for name in covariates.columns:
        if name not in df.columns:
            raise UserInputError(f"Explanatory variable '{name}' not found in data columns.")
        if name == time_col:
            raise UserInputError(f"Explanatory variable '{name}' must not equal time_col '{time_col}'.")
        if outcome_col is not None and name == outcome_col:
            raise UserInputError(f"Explanatory variable '{name}' must not equal outcome_col '{outcome_col}'.")
    for term in covariates.terms:
        if term not in df.columns:
            raise UserInputError(f"Explanatory variable '{term}' not found in data columns.")
        if not pd.api.types.is_numeric_dtype(df[term]):
            raise UserInputError(f"Explanatory variable '{term}' must be numeric, got dtype {df[term].dtype}.")


def drop_incomplete_rows(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop rows with NaN in any of the specified columns.

    Applied identically in all three partials before ``check_sample_size``
    so that thresholds apply to the analysed rows. The input is not
    modified; a copy is returned.
    """
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        return df.copy()
    return df.dropna(subset=existing_cols).copy()
