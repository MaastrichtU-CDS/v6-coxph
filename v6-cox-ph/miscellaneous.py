"""
Miscellaneous utilities for the Cox-PH algorithm.

This module contains the Pydantic models for the wire contract: user input
(``CoxPHInput``), the payload central sends to ``perform_iteration``
(``IterationInput``) and the partial results (``SummedZResult``,
``IterationResult``), which the partials build and central validates. It also
holds the explanatory-variable resolution (``parse_expl_vars``): the plain,
categorical and scored entries of ``expl_vars`` and the model terms they
expand to, plus the code key (``code_key``) every code comparison goes
through.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    ValidationError,
    field_validator,
    model_validator,
)
from vantage6.algorithm.tools.exceptions import UserInputError

# A category label becomes part of a model-term (result) key, so it is
# restricted to identifier-shaped text.
CATEGORY_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def format_validation_error(exc: ValidationError) -> str:
    """Format a pydantic error without echoing the offending input.

    Pydantic's default message embeds the input value; for wire payloads
    that value can be an aggregate array, and the message ends up in the
    container log the researcher reads. One line per error, no input, no URL.
    """
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(parts)


def code_key(value: Any) -> Optional[Union[float, str]]:
    """The comparison key of a covariate code.

    A missing value (``None``, NaN, an empty or whitespace-only string) maps
    to ``None``; a number, or a string that reads as one, maps to a float
    (so ``1``, ``"1"`` and ``1.0`` are the same code); any other string is
    compared stripped. This is the single extension point for code
    comparison: every function that maps codes takes it as ``key``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return stripped
        return None if math.isnan(parsed) else parsed
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return None if math.isnan(number) else number
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _check_code(code: Any, name: str, where: str) -> None:
    """Check that a listed code is a string or a number (not a bool).

    A code that is missing anyway (``None``, NaN, empty text) is allowed and
    ignored later, because its key is ``None``.
    """
    if code is None:
        return
    if isinstance(code, (bool, np.bool_)):
        raise ValueError(f"Covariate '{name}': {where} lists a boolean code {code!r}; codes are strings or numbers")
    if not isinstance(code, (str, int, float, np.integer, np.floating)):
        raise ValueError(f"Covariate '{name}': {where} lists a code that is not a string or a number: {code!r}")
    if isinstance(code, (float, np.floating)) and math.isnan(float(code)):
        return


def validate_covariate_spec(spec: Any, name: str) -> None:
    """Validate one categorical or scored covariate spec.

    A spec holds exactly one of ``categories`` or ``scores``, optionally
    ``missing``, and ``reference`` only with ``categories``; any other key
    is refused. ``categories`` maps at least two labels (each matching
    ``CATEGORY_LABEL_PATTERN``, because a label becomes part of a model-term
    name) to a non-empty list of codes, and ``reference`` is one of the
    labels. ``scores`` maps at least two codes to finite numbers. The
    messages echo only the name, labels and codes of the request.

    Raises
    ------
    ValueError
        On any malformed spec.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Covariate '{name}': the spec must be a mapping, got {type(spec).__name__}")
    allowed = {"categories", "scores", "missing", "reference"}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(f"Covariate '{name}': unknown spec keys {unknown}; allowed keys are {sorted(allowed)}")
    has_categories = "categories" in spec
    has_scores = "scores" in spec
    if has_categories == has_scores:
        raise ValueError(f"Covariate '{name}': the spec must hold exactly one of 'categories' or 'scores'")
    if "reference" in spec and not has_categories:
        raise ValueError(f"Covariate '{name}': 'reference' is only allowed with 'categories'")
    missing = spec.get("missing", [])
    if not isinstance(missing, list):
        raise ValueError(f"Covariate '{name}': 'missing' must be a list of codes")
    for code in missing:
        _check_code(code, name, "'missing'")
    if has_categories:
        categories = spec["categories"]
        if not isinstance(categories, dict) or len(categories) < 2:
            raise ValueError(f"Covariate '{name}': 'categories' must map at least two labels to their codes")
        for label, codes in categories.items():
            if not isinstance(label, str) or not CATEGORY_LABEL_PATTERN.match(label):
                raise ValueError(
                    f"Covariate '{name}': category label {label!r} is invalid; labels match "
                    f"^[A-Za-z0-9_][A-Za-z0-9_.-]*$ (a label becomes part of a model-term name)"
                )
            if not isinstance(codes, list) or len(codes) == 0:
                raise ValueError(f"Covariate '{name}': category '{label}' must list at least one code")
            for code in codes:
                _check_code(code, name, f"category '{label}'")
        reference = spec.get("reference")
        if not isinstance(reference, str) or reference not in categories:
            raise ValueError(f"Covariate '{name}': 'reference' must be one of the category labels {sorted(categories)}")
    else:
        scores = spec["scores"]
        if not isinstance(scores, dict) or len(scores) < 2:
            raise ValueError(f"Covariate '{name}': 'scores' must map at least two codes to numbers")
        for code, score in scores.items():
            _check_code(code, name, "'scores'")
            if isinstance(score, (bool, np.bool_)) or not isinstance(score, (int, float, np.integer, np.floating)):
                raise ValueError(f"Covariate '{name}': the score of code {code!r} must be a number")
            if not math.isfinite(float(score)):
                raise ValueError(f"Covariate '{name}': the score of code {code!r} must be finite")


@dataclass(frozen=True)
class CategoricalCovariate:
    """A categorical explanatory variable: one 0/1 model term per non-reference category."""

    name: str
    reference: str
    categories: Dict[str, List[Any]]
    missing: List[Any]


@dataclass(frozen=True)
class ScoredCovariate:
    """A scored (ordinal) explanatory variable: one numeric model term."""

    name: str
    scores: Dict[Any, float]
    missing: List[Any]


@dataclass(frozen=True)
class ResolvedCovariates:
    """The resolved explanatory variables of a request.

    ``columns`` are the source columns (the entry names, in order); ``terms``
    are the model terms they expand to — one per plain or scored variable,
    one per non-reference category of a categorical variable, in entry
    order. ``plain``, ``categorical`` and ``scored`` hold the entries by
    kind.
    """

    columns: List[str]
    terms: List[str]
    plain: List[str]
    categorical: List[CategoricalCovariate]
    scored: List[ScoredCovariate]


def _entry_name(entry: Union[str, Dict[str, Any]]) -> str:
    """The variable name of one ``expl_vars`` entry."""
    if isinstance(entry, str):
        return entry
    return next(iter(entry))


def build_code_map(spec: Union[CategoricalCovariate, ScoredCovariate], key: Callable = code_key) -> Dict[Any, Any]:
    """Build the code map of a categorical or scored covariate.

    Maps the code key of every listed code to its category label
    (``CategoricalCovariate``) or score (``ScoredCovariate``); listed
    ``missing`` codes map to ``None``. A listed code whose key is ``None`` is
    ignored, because it is missing anyway.

    Raises
    ------
    ValueError
        When two listed codes collide under ``key`` — a spec error that
        depends on the request only, never on data.
    """
    pairs: List[Tuple[Any, Any]] = []
    if isinstance(spec, CategoricalCovariate):
        for label, codes in spec.categories.items():
            for code in codes:
                pairs.append((code, label))
    elif isinstance(spec, ScoredCovariate):
        for code, score in spec.scores.items():
            pairs.append((code, score))
    else:
        raise TypeError(f"Cannot build a code map for {type(spec).__name__}")
    for code in spec.missing:
        pairs.append((code, None))
    code_map: Dict[Any, Any] = {}
    seen: Dict[Any, Any] = {}
    for code, mapped in pairs:
        code_key_value = key(code)
        if code_key_value is None:
            continue
        if code_key_value in seen:
            raise ValueError(
                f"Covariate '{spec.name}': the codes {seen[code_key_value]!r} and {code!r} are the same "
                f"under the code key; list each code once"
            )
        seen[code_key_value] = code
        code_map[code_key_value] = mapped
    return code_map


def parse_expl_vars(expl_vars: List[Union[str, Dict[str, Any]]]) -> ResolvedCovariates:
    """Resolve the explanatory variables of a request into model terms.

    Each entry is a plain column name or a single-entry mapping of a column
    name to a categorical or scored spec (validated by
    ``validate_covariate_spec``). Names must be unique across entries; the
    model terms must be unique and differ from every entry's name. Codes are
    compared through ``code_key``; two listed codes that collide under it are
    a spec error, raised here when the code map is built.

    Raises
    ------
    ValueError
        On any malformed entry, duplicate name, clashing term or colliding
        code.
    """
    if not isinstance(expl_vars, list) or len(expl_vars) == 0:
        raise ValueError("At least one explanatory variable must be specified")
    columns: List[str] = []
    terms: List[str] = []
    categorical_terms: List[str] = []
    plain: List[str] = []
    categorical: List[CategoricalCovariate] = []
    scored: List[ScoredCovariate] = []
    for entry in expl_vars:
        if isinstance(entry, str):
            name = entry.strip()
            if not name:
                raise ValueError(f"Invalid variable name: {entry!r}")
            columns.append(name)
            terms.append(name)
            plain.append(name)
            continue
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                "Each explanatory variable must be a name or a mapping of exactly one name to a spec, " f"got {entry!r}"
            )
        name, spec = next(iter(entry.items()))
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid variable name: {name!r}")
        name = name.strip()
        validate_covariate_spec(spec, name)
        columns.append(name)
        if "categories" in spec:
            categorical_entry = CategoricalCovariate(
                name=name,
                reference=spec["reference"],
                categories={label: list(codes) for label, codes in spec["categories"].items()},
                missing=list(spec.get("missing", [])),
            )
            categorical.append(categorical_entry)
            for label in categorical_entry.categories:
                if label != categorical_entry.reference:
                    term = f"{name}[{label}]"
                    terms.append(term)
                    categorical_terms.append(term)
            # A spec error (two listed codes colliding under the code key) is
            # raised here, when the map is built; it depends on the request
            # only.
            build_code_map(categorical_entry)
        else:
            scored_entry = ScoredCovariate(
                name=name,
                scores={code: float(score) for code, score in spec["scores"].items()},
                missing=list(spec.get("missing", [])),
            )
            scored.append(scored_entry)
            terms.append(name)
            build_code_map(scored_entry)
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise ValueError(f"expl_vars contains duplicate variables: {duplicates}")
    # A categorical term '<name>[<label>]' must differ from every entry's
    # name (a plain or scored term is its own entry's name by design).
    clash = sorted(set(categorical_terms) & set(columns))
    if clash:
        raise ValueError(
            f"Model terms {clash} clash with explanatory variable names {columns}; "
            f"a term '<name>[<label>]' must differ from every variable name"
        )
    term_duplicates = sorted({term for term in terms if terms.count(term) > 1})
    if term_duplicates:
        raise ValueError(f"expl_vars produces duplicate model terms: {term_duplicates}")
    return ResolvedCovariates(columns=columns, terms=terms, plain=plain, categorical=categorical, scored=scored)


def expand_covariate_profile(
    profile: Dict[str, Union[float, str]],
    covariates: ResolvedCovariates,
) -> Dict[str, float]:
    """Expand a covariate profile into model-term values.

    A plain or scored variable takes its number; a categorical variable's
    label expands to the 1/0 values of that variable's terms (the reference
    label: all 0). Variables left out stay at the pooled event-case centre
    (``survival_curves`` fills them in).
    """
    expanded: Dict[str, float] = {}
    categorical = {entry.name: entry for entry in covariates.categorical}
    for name, value in profile.items():
        if name in categorical:
            entry = categorical[name]
            for label in entry.categories:
                if label == entry.reference:
                    continue
                expanded[f"{name}[{label}]"] = 1.0 if label == value else 0.0
        else:
            expanded[name] = float(value)
    return expanded


class CoxPHInput(BaseModel):
    """
    Pydantic model for validating Cox-PH algorithm input parameters.

    This validates all user-supplied parameters that travel over the wire
    between central and partial functions.
    """

    time_col: str = Field(..., description="Name of the column containing time data")
    outcome_col: str = Field(..., description="Name of the column containing outcome/event data")
    expl_vars: List[Union[str, Dict[str, Any]]] = Field(
        ...,
        description=(
            "List of explanatory variables: plain column names, or single-entry "
            "mappings of a column name to a categorical spec (reference and "
            "categories) or a scored spec (scores, optional missing)."
        ),
    )
    organization_ids: Optional[List[int]] = Field(
        default=None,
        description=("List of organisation IDs to include. If None, all " "organisations are used."),
    )
    covariate_profiles: Optional[List[Dict[str, Union[float, str]]]] = Field(
        default=None,
        description=(
            "Optional covariate profiles for survival curves. Each entry maps "
            "explanatory variable names to values: a number for a plain or "
            "scored variable, a category label for a categorical variable. "
            "Variables left out are taken at the pooled event-case mean (the "
            "baseline profile)."
        ),
    )

    @field_validator("time_col", "outcome_col")
    @classmethod
    def validate_column_names(cls, v: str) -> str:
        """Validate that column names are non-empty strings."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Column names must be non-empty strings")
        return v.strip()

    @field_validator("expl_vars")
    @classmethod
    def validate_expl_vars(cls, v: List[Union[str, Dict[str, Any]]]) -> List[Union[str, Dict[str, Any]]]:
        """Validate the explanatory-variable entries (names or specs)."""
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("At least one explanatory variable must be specified")
        normalised: List[Union[str, Dict[str, Any]]] = []
        for entry in v:
            if isinstance(entry, str):
                if not entry.strip():
                    raise ValueError(f"Invalid variable name: {entry}")
                normalised.append(entry.strip())
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(
                        "Each explanatory variable must be a name or a mapping of exactly one name to a spec"
                    )
                name, spec = next(iter(entry.items()))
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Invalid variable name: {name}")
                validate_covariate_spec(spec, name.strip())
                normalised.append({name.strip(): spec})
            else:
                raise ValueError(f"Invalid explanatory variable entry: {entry!r}")
        return normalised

    @field_validator("organization_ids")
    @classmethod
    def validate_organization_ids(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Validate organisation IDs if provided."""
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("organization_ids must be a list of integers")
        for org_id in v:
            if not isinstance(org_id, int) or org_id < 0:
                raise ValueError(f"Invalid organisation ID: {org_id}. " "Must be a non-negative integer.")
        return list(set(v))

    @model_validator(mode="after")
    def validate_column_semantics(self) -> "CoxPHInput":
        """The time, outcome and explanatory columns must be distinct.

        The messages echo only the user-supplied column names. The
        model-term rules (unique terms, distinct from every entry's name)
        are enforced by ``parse_expl_vars``; a model term must also differ
        from the time and outcome columns, because the node encodes the
        term into a column of that name.
        """
        if self.time_col == self.outcome_col:
            raise ValueError(f"time_col and outcome_col must be different columns, both are '{self.time_col}'")
        names = [_entry_name(entry) for entry in self.expl_vars]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"expl_vars contains duplicate variables: {duplicates}")
        clash = [name for name in names if name in (self.time_col, self.outcome_col)]
        if clash:
            raise ValueError(
                f"expl_vars must not name the time or outcome column; found {clash} "
                f"(time_col='{self.time_col}', outcome_col='{self.outcome_col}')"
            )
        covariates = parse_expl_vars(self.expl_vars)
        term_clash = [term for term in covariates.terms if term in (self.time_col, self.outcome_col)]
        if term_clash:
            raise ValueError(
                f"Model terms must not equal the time or outcome column; found {term_clash} "
                f"(time_col='{self.time_col}', outcome_col='{self.outcome_col}')"
            )
        return self

    @model_validator(mode="after")
    def validate_covariate_profiles(self) -> "CoxPHInput":
        """Covariate profiles may only name known variables with valid values.

        A plain or scored variable takes a number; a categorical variable
        takes one of its labels.
        """
        if self.covariate_profiles is None:
            return self
        covariates = parse_expl_vars(self.expl_vars)
        categorical = {entry.name: entry for entry in covariates.categorical}
        for i, profile in enumerate(self.covariate_profiles):
            if not isinstance(profile, dict):
                raise ValueError(f"Covariate profile {i} must be a mapping of variable names to values")
            unknown = [var for var in profile if var not in covariates.columns]
            if unknown:
                raise ValueError(
                    f"Covariate profile {i} names unknown variables {unknown}; "
                    f"known explanatory variables are {covariates.columns}"
                )
            for var, value in profile.items():
                if var in categorical:
                    labels = list(categorical[var].categories)
                    if value not in labels:
                        raise ValueError(
                            f"Covariate profile {i}: variable '{var}' is categorical; its value must be "
                            f"one of its labels {labels}, got {value!r}"
                        )
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Covariate profile {i}: variable '{var}' takes a number, got {value!r}")
        return self

    @property
    def model_terms(self) -> List[str]:
        """The model terms of the explanatory variables (``parse_expl_vars``).

        The terms are the names in the result's ``model``, the order of
        ``covariance``, ``beta``, ``centre`` and ``scale``, and the keys of
        the partials' ``sum``/``sum_squares``.
        """
        return parse_expl_vars(self.expl_vars).terms

    @property
    def expanded_covariate_profiles(self) -> List[Dict[str, float]]:
        """The covariate profiles expanded to model-term values."""
        covariates = parse_expl_vars(self.expl_vars)
        return [expand_covariate_profile(profile, covariates) for profile in (self.covariate_profiles or [])]


def validate_coxph_input(
    time_col: str,
    outcome_col: str,
    expl_vars: List[Union[str, Dict[str, Any]]],
    organization_ids: Optional[List[int]] = None,
    covariate_profiles: Optional[List[Dict[str, Union[float, str]]]] = None,
) -> CoxPHInput:
    """
    Validate Cox-PH algorithm input parameters.

    Parameters
    ----------
    time_col : str
        Name of the time column.
    outcome_col : str
        Name of the outcome column.
    expl_vars : list
        Explanatory variables: plain column names, or single-entry mappings
        of a column name to a categorical or scored spec.
    organization_ids : Optional[List[int]]
        List of organisation IDs to include.
    covariate_profiles : Optional[List[Dict[str, Union[float, str]]]]
        Optional covariate profiles for survival curves.

    Returns
    -------
    CoxPHInput
        Validated input model.

    Raises
    ------
    UserInputError
        If input validation fails.
    """
    try:
        return CoxPHInput(
            time_col=time_col,
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            organization_ids=organization_ids,
            covariate_profiles=covariate_profiles,
        )
    except ValidationError as e:
        # The formatted message never echoes the offending input: pydantic's
        # default message embeds it, and the message ends up in the container
        # log the researcher reads.
        raise UserInputError(f"Invalid Cox-PH input: {format_validation_error(e)}") from None


class IterationInput(BaseModel):
    """
    Wire payload ``central`` sends to ``perform_iteration`` each round-trip.

    Validated on the node before any aggregate is computed. The expected
    lengths (one entry per model term) and the node's privacy settings are
    not part of the payload: they arrive via the validation ``context``
    (``expl_vars`` — the model terms — and ``settings``).

    Structure is validated by pydantic; the numeric checks (finiteness,
    positivity, monotonicity, the bin grid) run through numpy on the plain
    lists, which is faster than element-wise validation and adds no safety.
    """

    beta: List[float]
    centre: List[float]
    scale: List[float]
    unique_time_events: List[float]

    @field_validator("beta", "centre", "scale")
    @classmethod
    def _finite_vector(cls, v: List[float], info: ValidationInfo) -> List[float]:
        if not np.all(np.isfinite(np.asarray(v, dtype=float))):
            raise ValueError(f"{info.field_name} contains non-finite values (NaN or inf).")
        return v

    @field_validator("scale")
    @classmethod
    def _positive_scale(cls, v: List[float]) -> List[float]:
        if np.any(np.asarray(v, dtype=float) <= 0):
            raise ValueError("scale must be strictly positive for every explanatory variable.")
        return v

    @field_validator("unique_time_events")
    @classmethod
    def _valid_grid(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("unique_time_events must not be empty.")
        grid = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(grid)):
            raise ValueError("unique_time_events contains non-finite values.")
        if not np.all(np.diff(grid) > 0):
            # Strictly increasing covers both the sortedness and the
            # no-duplicates rules.
            raise ValueError("unique_time_events must be strictly increasing (sorted, without duplicates).")
        return v

    @model_validator(mode="after")
    def _lengths_and_bin_grid(self, info: ValidationInfo) -> "IterationInput":
        context = info.context or {}
        expl_vars = context.get("expl_vars", [])
        for name in ("beta", "centre", "scale"):
            vector = getattr(self, name)
            if len(vector) != len(expl_vars):
                raise ValueError(
                    f"{name} must have length {len(expl_vars)} (one per explanatory "
                    f"variable), got shape {np.asarray(vector).shape}"
                )
        settings = context.get("settings")
        width = getattr(settings, "time_bin_width", None)
        if width:
            grid = np.asarray(self.unique_time_events, dtype=float)
            binned = np.floor(grid / width) * width
            off_grid = ~np.isclose(binned, grid)
            if np.any(off_grid):
                raise ValueError(
                    f"unique_time_events contains time {grid[off_grid][0]} that is "
                    f"not on the bin grid (width={width})."
                )
        return self


class SummedZResult(BaseModel):
    """
    Result of the ``compute_summed_z`` sub-task.

    Built on the node (no context) and returned as ``model_dump()``;
    validated by central with the runtime context (``expected_ids``,
    ``expl_vars`` — the model terms —, ``time_col``). The context checks only
    run when the context provides them. ``extra="forbid"``: central and the
    partials ship in one image, so an unexpected key is contract drift, not
    forward compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    organization_id: int = Field(strict=True)
    sum: Dict[str, float]
    sum_squares: Dict[str, float]
    times: Dict[str, Any]
    privacy_settings: Dict[str, Any]

    @model_validator(mode="after")
    def _matches_request(self, info: ValidationInfo) -> "SummedZResult":
        context = info.context or {}
        expected_ids = context.get("expected_ids")
        if expected_ids is not None and self.organization_id not in expected_ids:
            raise ValueError(f"organization_id {self.organization_id!r} is not one of the dispatched organisations.")
        expl_vars = context.get("expl_vars")
        if expl_vars is not None:
            for name in ("sum", "sum_squares"):
                missing = [v for v in expl_vars if v not in getattr(self, name)]
                if missing:
                    raise ValueError(f"'{name}' missing variables {missing}")
        time_col = context.get("time_col")
        if time_col is not None:
            times_df = pd.DataFrame.from_dict(self.times)
            if time_col not in times_df.columns or "freq" not in times_df.columns:
                raise ValueError(f"'times' must have columns '{time_col}' and 'freq', got {list(times_df.columns)}")
            if len(times_df) > 0:
                freqs = times_df["freq"].to_numpy()
                if not np.all(np.isfinite(freqs)) or np.any(freqs < 0):
                    raise ValueError("'freq' contains non-finite or negative values")
        return self


class IterationResult(BaseModel):
    """
    Result of the ``perform_iteration`` sub-task.

    Built on the node (no context) and returned as ``model_dump()``;
    validated by central with the runtime context (``expected_ids``,
    ``n_times``, ``n_covs``). The context checks only run when the context
    provides them. ``extra="forbid"`` as on ``SummedZResult``.
    """

    model_config = ConfigDict(extra="forbid")

    organization_id: int = Field(strict=True)
    agg1: List
    agg2: List
    agg3: List

    @model_validator(mode="after")
    def _shapes_and_finiteness(self, info: ValidationInfo) -> "IterationResult":
        context = info.context or {}
        expected_ids = context.get("expected_ids")
        if expected_ids is not None and self.organization_id not in expected_ids:
            raise ValueError(f"organization_id {self.organization_id!r} is not one of the dispatched organisations.")
        n_times = context.get("n_times")
        n_covs = context.get("n_covs")
        if n_times is None or n_covs is None:
            return self
        agg1 = np.asarray(self.agg1, dtype=float)
        if agg1.ndim != 1 or len(agg1) != n_times:
            raise ValueError(f"agg1 has length {len(agg1)}, expected {n_times}")
        if not np.all(np.isfinite(agg1)):
            raise ValueError("agg1 contains non-finite values")
        agg2 = np.asarray(self.agg2, dtype=float)
        if agg2.shape != (n_times, n_covs):
            raise ValueError(f"agg2 has shape {agg2.shape}, expected ({n_times}, {n_covs})")
        if not np.all(np.isfinite(agg2)):
            raise ValueError("agg2 contains non-finite values")
        agg3 = np.asarray(self.agg3, dtype=float)
        if agg3.shape != (n_times, n_covs, n_covs):
            raise ValueError(f"agg3 has shape {agg3.shape}, expected ({n_times}, {n_covs}, {n_covs})")
        # Stricter than 2.0.0, which did not check agg3 for finiteness.
        if not np.all(np.isfinite(agg3)):
            raise ValueError("agg3 contains non-finite values")
        return self
