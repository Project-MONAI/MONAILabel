# Verification

Use disposable workspaces for mutation and viewer tests. Never run a second server against an active workspace. The standard lint, type and test commands are in [AGENTS.md](../AGENTS.md#development).

## Golden prompts

The eight [Quickstart prompts](../README.md#try-the-full-workflow) come from [spleen.json](../examples/prompts/spleen.json). The runner verifies imports, scoped reviews, batch annotation, named model creation, training and comparison against independent fixed references.

```bash
uv run python examples/render_golden_prompts.py --check
# Deterministic tools and tiny images; no GPU, network or medical weights:
uv run python examples/verify_spleen_workflow.py --output /tmp/spleen-fixture.json
# Test all eight prompts with either managed local conversation model:
uv run python examples/verify_spleen_workflow.py --coordinator 4b --output /tmp/spleen-4b.json
uv run python examples/verify_spleen_workflow.py --coordinator 9b --output /tmp/spleen-9b.json
# Real VISTA3D and Decathlon; connect your existing conversation endpoint:
uv run python examples/verify_spleen_workflow.py --real \
  --coordinator-url http://127.0.0.1:8001/v1 \
  --coordinator-model YOUR_SERVED_MODEL \
  --cache-dir workspace/.cache/datasets \
  --output /tmp/spleen-real.json
```

The default runner creates and removes a temporary workspace. Supply `--workspace /path/to/empty-directory` to retain results for inspection. `--cache-dir` reuses dataset downloads; `MONAILABEL_MODELS_DIR` selects a model cache. Omit `--real` to test a conversation endpoint against deterministic inference/training fixtures. Fixture scores verify software behavior, not medical quality. Reports include the instruction revision, loaded skills and tool-call traces. Use `--prompts /path/to/definition.json` to exercise paraphrases and different model names while retaining the workflow's assertions. Repeat runs in fresh workspaces to check consistency; one passing run does not establish reliability.

The real run imports 41 cases, reserves 9 labeled references, annotates 5 of the remaining 32 images, fine-tunes a derived model, and scores it and the unchanged base on identical references. It checks image membership, annotation revisions, model lineage, logs and reports. Reviews are explicitly accepted by the test prompts; this does not replace human inspection in normal use.

Additional [radiology](../examples/prompts/radiology.json) and [pathology](../examples/prompts/pathology.json) definitions exercise viewer actions, model choices and continued U-Net learning:

```bash
uv run python examples/verify_golden_stories.py --story both --output /tmp/golden-stories.json
uv run python examples/verify_golden_stories.py --coordinator lightning --story both --output /tmp/lightning-stories.json
# Tool selection only; does not execute operational tools:
uv run python examples/evaluate_coordinators.py --variant lightning --output /tmp/lightning-prompts.json
node --test tests/web/*.mjs
```

## Native viewers

Check Slicer, QuPath and OHIF on disposable images with asymmetric geometry. Verify source orientation, selected-slice scope, class colors, editable hints, stale-revision rejection, submission and corrected review. Preserve user drafts. Headless browser checks do not establish native desktop or physical mobile-device compatibility.

Use `uv run python examples/vista3d_smoke.py --help` for the VISTA3D GPU smoke check. SAM contracts and viewer transfers are exercised in `tests/test_sam.py` and the spatial-hint tests. These exercise runtime and transfer behavior; they are not clinical benchmarks. Remaining capabilities and platform gaps are listed in the [roadmap](roadmap.md).
