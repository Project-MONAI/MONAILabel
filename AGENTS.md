# Working on MONAI Label

Read [README.md](README.md) for setup and [docs/implementation.md](docs/implementation.md) for package boundaries. Keep changes focused and documentation about current behavior.

## Design

- The web workspace manages projects, datasets, models, reviews and learning. Slicer, QuPath and OHIF share its backend and assistant.
- Use typed tools for chat actions. Keep domain contracts independent of HTTP, storage and model vendors; use replaceable provider/trainer interfaces.
- Support multiple models and structures. Inference and training are independent. Preserve base weights; training publishes derived versions with lineage.
- Ordinary datasets stay shared. Each model owns its stable training/validation split. Independent evaluation-only sets are excluded from every model's training.
- Preserve source geometry, patient/slide grouping, immutable annotation revisions and dataset snapshots. Check revisions before applying changes. Evaluation uses accepted, held-out references and rejects known training overlap.
- Keep defaults simple and advanced settings optional. Distinguish implemented features from [planned work](docs/roadmap.md).

## Development

- Use the uv workspace. Providers and clients must not import server code.
- Protect user annotations and viewer drafts. Use disposable workspaces for mutation tests; never run a second server against an active workspace.
- Keep credentials out of code, chat and docs. Use environment references or the credential store. Do not invoke paid models without a specific authorized need.
- Keep README commands runnable and the synthetic demo clearly identified.
- Do not hard-wrap Markdown prose; let the reader or editor wrap it. Keep a blank line between example prompts.
- Update the relevant guide when behavior changes. Avoid session handoffs, process IDs, machine-specific state and dated test-count summaries in project docs.

For implementation changes, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy packages
uv run pytest
```

Run relevant browser/viewer checks for UI changes. Golden prompt documentation is checked with `uv run python examples/render_golden_prompts.py --check`.
