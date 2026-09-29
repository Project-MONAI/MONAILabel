# Architecture

## Packages

Each `packages/` directory is a uv workspace project with its own dependencies. The root owns the shared lockfile and development tools. Releases bundle these projects into the single `monailabel` distribution.

| Package | Responsibility |
| --- | --- |
| `monailabel` | Launcher and public release metadata |
| `core` | Domain records, geometry, provider/trainer interfaces |
| `providers` | Local baselines, hosted annotation and conversation adapters |
| `monai` | U-Net, nnU-Net v2 and VISTA3D inference/training |
| `totalsegmentator` | CT/MRI inference and fine-tuning in isolated workers |
| `sam` | SAM 2.1 and MedSAM2 adapters |
| `nninteractive` | Prompted CT/MRI inference in a pinned worker environment |
| `sam-runtime` | Pinned upstream SAM inference source |
| `dicom` | DICOMweb import, decoding and viewing conversion |
| `viewers` | Slicer, QuPath, OHIF and CVAT provisioning/adapters |
| `server` | Services, storage, jobs, authentication, API and web console |
| `client` | HTTP client, CLI and synthetic demo |

`core` has no HTTP, storage or vendor dependencies. Providers and clients must not import server code. Python packages share the `monailabel` namespace without a common `__init__.py`; the SAM runtime uses upstream `sam2` and `efficient_track_anything` namespaces.

Within `server`, feature services own lifecycle and revision rules; routes handle authentication and transport. `assistant_tools/` adapts typed actions to those services. Keep viewer provisioning in `viewers`, catalog discovery in `providers/catalog/`, and DICOM decoding in `dicom`. Providers never access workspace storage.

## Data and learning

- Images use `H×W×C` or `I×J×K×1` arrays. Saved masks use uint8 project class IDs on the source grid, in C-order. Training masks use signed integers; `-1` excludes unreviewed pixels.
- Proposals record geometry, model and base revision. Applying or submitting checks that revision and any intervening viewer draft changes.
- Slice inference restores source orientation before merging. Region inference updates only its crop. Failed or cancelled multi-part inference publishes no partial proposal.
- Annotation revisions and review decisions are immutable. Corrected acceptance saves the revision and decision atomically. Region and video-range revisions retain independent review decisions.
- Each model owns a stable patient/slide/procedure split over shared datasets. Snapshots freeze image, annotation and review revisions. Evaluation-only groups are excluded from every model's training.
- Evaluation requires accepted references and rejects known training overlap. Training preserves base weights and publishes derived versions with lineage; model promotion is explicit.

Video records retain original files and presentation timestamps. Tracks use zero-based source frames and pixel-edge coordinates. CVAT drafts, submitted tracks and lossless proposal masks are stored separately. Accepted polygon ranges provide segmentation samples; unreviewed pixels are excluded from loss. See [video contracts](video.md#api-and-data-retention) and [learning rules](workflows.md#train-a-model).

## Services and storage

One server process owns a workspace lock. SQLite transactions publish related records atomically; arrays, files and checkpoints are content-addressed. Background jobs support cooperative cancellation. Restart marks unfinished jobs interrupted. Idempotency keys require matching project, operation and payload.

The default workspace is `workspace/`, with downloads under `.cache/`. Credentials are encrypted using `secrets.key`; back it up with the database. Passwords use salted scrypt and stored session tokens are hashed.

Deletion checks active jobs and saved learning/reference dependencies. Project deletion retains shared caches and external viewer data. Unreferenced artifacts are collected before workers start.

## API and clients

The running server exposes `/docs` and `/openapi.json`. Browser cookies and bearer tokens use the same project authorization. UI and assistant tools call the same services. The coordinator has no shell or arbitrary-code tool.

Slicer/QuPath run locally or in account-owned browser desktops. `viewers` owns provisioning and display processes; `server` owns sessions, credentials and authenticated display transport. OHIF/CVAT use browser sessions. Viewer adapters preserve native geometry and drafts; backend providers execute models.

Interactive models declare supported input types and output scopes. Slicer, OHIF, QuPath and CVAT build their controls from those capabilities and retain hints per image and target; video inputs also belong to an object and frame. Typed assistant actions start, stop or edit hints. Viewer annotation skills are filtered by the source's image, volume or video kind. Update calls the inference service directly, with the same revision and draft checks as assistant-triggered inference.

See [provider contracts](providers.md), [viewer setup](viewers.md), [release checks](releasing.md) and [development checks](../AGENTS.md#development).
