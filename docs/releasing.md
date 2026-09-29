# Build and release

## Prepare a release

Publish only `monailabel`. The build combines the workspace components into one wheel and one self-contained source archive, including their dependencies and license notices.

1. Set the same version in the root and every package's `pyproject.toml`, including internal dependency pins. Update `MONAILABEL_VERSION` in the Dockerfile and run `uv lock`.
2. Include each package's README, license files and required [third-party notices](../THIRD_PARTY_NOTICES.md#updating-third-party-material).
3. Run the [development checks](../AGENTS.md#development), then build and test the distributions:

```bash
uv run --no-project python scripts/build_package.py
uv run python scripts/check_packages.py
uv run python scripts/check_packages.py --python 3.13
uvx twine check --strict dist/*
```

To install the local release in a fresh environment:

```bash
python -m pip install -U --pre --find-links=/absolute/path/to/MONAILabel/dist monailabel
```

See [Docker build checks](docker.md#verify-a-build) for image verification.

## One-time registry configuration

- Create GitHub environments `pypi` and `testpypi` in `Project-MONAI/MONAILabel`.
- Configure a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for `monailabel` on each registry: repository `Project-MONAI/MONAILabel`, workflow `release.yml`, and the matching environment. Use the existing `monailabel` project on PyPI.
- Grant Docker Hub account `projectmonai` access to `projectmonai/monailabel` and set the repository secret `DOCKER_PW`.

## Workflows

| Workflow | Action |
| --- | --- |
| Checks / Package | Development checks, wheels/sdists, metadata and license validation, fresh pip installs on Linux x86_64/ARM64 with Python 3.12/3.13. Produces `python-dist`. |
| Docker | Tests native amd64/arm64 images for pull requests, schedules and releases. Publishes multi-platform images with SBOM and provenance. |
| Release | Requires tag `v<version>`, waits for Checks, publishes `monailabel`, then publishes Docker images. Stable releases update `latest`; scheduled builds use `dev`. |

Dispatch **Release** to TestPyPI first; Docker publication defaults off. Download `python-dist` to inspect the built packages. Publish a GitHub release with the matching tag when ready for PyPI and Docker Hub. Forks do not publish.
