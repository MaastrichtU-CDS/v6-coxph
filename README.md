# v6-cox-ph

A federated Cox proportional hazards model (Breslow ties) for
[vantage6](https://vantage6.ai) 4.14–4.15. The model is fitted by
Newton–Raphson across the data stations: only aggregated quantities leave a
node, and every node checks a sample-size threshold and further privacy
guards before it shares anything.

This is the vanilla algorithm. Project-specific features (row selection,
recoding, other data sources) belong in a fork or a stacked branch; see
*Building on this algorithm* in [`AGENTS.md`](AGENTS.md).

## Documentation

The full documentation (usage, privacy, implementation and validation) is in
the [repository wiki](../../wiki). The wiki is published from `docs/` on the
default branch, and every page states the release or development version it
describes. The documentation of this branch is in [`docs/coxph`](docs/coxph).

## Quick start

Call `central`; it dispatches the node-side partials itself and returns the
fitted model as JSON.

```python
task = client.task.create(
    collaboration=1,
    organizations=[1],  # the organisation that runs central
    name="Cox proportional hazards",
    image="ghcr.io/<owner>/<repository>-<branch>:<tag>",
    input_={
        "method": "central",
        "kwargs": {
            "time_col": "overall_survival_in_days",
            "outcome_col": "event_overall_survival",  # 1 = event, 0 = censored
            "expl_vars": ["age", {"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}}],
            "organization_ids": [2, 3],  # the data stations to include
        },
    },
    databases=[{"label": "default"}],
)
result = client.wait_for_results(task_id=task["id"])
```

The input arguments, the output fields and the survival curves are described
on the *Usage* page of the wiki.

## Node configuration

Data-station managers tune the privacy guards through the node's
`algorithm_env`, for example `SAMPLE_SIZE_THRESHOLD` (default 10) and the
`COXPH_PRIVACY_LEVEL` preset (`minimal`, `standard` or `strict`). The *Privacy*
page of the wiki lists every setting with its default and what it protects
against.

## Development

Python 3.10 or later (the image runs Python 3.12):

```bash
pip install -e .[dev,lint]
pytest -m unit                # no Docker needed
pytest tests/integration      # needs Docker and the v6 CLI
```

The lint, type-check and Docker commands that CI runs are listed in
[`AGENTS.md`](AGENTS.md).

## Releases

Releases are git tags and follow semantic versioning; see
[`CHANGELOG.md`](CHANGELOG.md). A tag publishes the image
`ghcr.io/<owner>/<repository>-<branch>:<tag>`, where `<branch>` is the branch
whose tip is the tagged commit.

## Licence

Apache License 2.0; see [`LICENSE`](LICENSE).
