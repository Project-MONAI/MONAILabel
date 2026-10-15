# Staged demo installation

This branch provides `scripts/demo_install.sh` for preparing a dedicated demo host without running the full workstation setup. Installation is not public-demo readiness. See [private demo controls](demo-controls.md) for the opt-in account deadline, research telemetry and dataset cache-only foundation, and the remaining admission, resource, cleanup, viewer and network isolation gates.

## Approval boundaries

The default command only prints a plan. It does not download, install, create accounts, provision viewers or start services:

```bash
bash scripts/demo_install.sh
```

The `system` mode prints administrator commands using the platform-specific prerequisite list from `setup.sh`. It never executes them and rejects `--apply`:

```bash
bash scripts/demo_install.sh system
```

Review the printed APT commands before running them in an administrator's own terminal. They install system libraries and tools including Git, curl and FFmpeg. They do not install Docker or change group memberships. Package-manager maintainer scripts can affect existing system services; inspect the APT proposal before confirming it. Keep the demo account without sudo and do not execute a service-user-writable installation script as root. Docker-group access is a separate root-equivalent permission, described below.

## Combined private-host administrator step

For a dedicated Ubuntu 22.04 x86_64 machine, `scripts/demo_host_bootstrap.sh` combines system packages, Docker/Compose, NVIDIA Container Toolkit and trusted service-account permissions. Its default is read-only. It does not source repository code as root. Review the script, copy it to `/root/monailabel-demo-bootstrap.sh` with root ownership and mode `0700`, verify its checksum against the reviewed source, and run the protected copy with `--apply --allow-docker-root-access`.

The explicit flag accepts root-equivalent Docker access for the existing `demo` account. Do not use that account as an anonymous visitor login. The script enables the account's systemd user services after logout, but grants no sudo rights. It preserves the working GPU driver, refuses conflicting container packages or a restart with running containers, backs up an existing Docker daemon configuration, and retains its bootstrap staging. An interrupted bootstrap may have installed some packages; inspect its output before retrying. It does not roll back package installation.

This follows the official [Docker Ubuntu installation](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA container runtime setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Docker uses its default system storage directories; model and application caches use the separately configured installation root. The daemon creates its normal bridge and networking rules, but the bootstrap does not publish application ports, modify SSH or configure external access. Package versions are selected from the official repositories, not pinned into a reproducible host image. The GPU verification downloads a small Ubuntu container and runs `nvidia-smi` without container networking.

## User-owned environment

After those prerequisites are approved and installed, review the environment phase separately. Run it as the dedicated `demo` account, without sudo, with an existing demo-owned installation root:

```bash
bash scripts/demo_install.sh environment --root /opt/monailabel-demo

# Only after separate approval; this downloads and installs software.
bash scripts/demo_install.sh environment --root /opt/monailabel-demo --apply
```

This installs uv 0.11.8 using its versioned HTTPS installer, managed Python 3.12, the workspace's locked dependencies including development tools, and Node/Corepack when an adequate installation is not already available. The Node archive and checksums come from the existing setup helper. Dependency packages, including CUDA/PyTorch libraries, can be large. This phase does not intentionally download model weights or datasets, invoke models, run tests or provision any viewer. Third-party package builds execute their own build code as the unprivileged user.

The wrapper sources `setup.sh` to reuse selected helpers; it does not invoke its `main`, Docker, GPU-container or viewer-provisioning functions. The uv installer uses its [unmanaged installation option](https://docs.astral.sh/uv/reference/installer/#unmanaged-installations), which avoids profile modification and self-updates. Package versions follow `uv.lock`; the selected Python 3.12 patch release and APT package versions can change between installations. This is not a fully immutable environment image.

## Filesystem layout

Keep `src/` for the source snapshot, `env/` for the Python environment, `tools/` for managed tools, `cache/` for downloads, `workspace/` for future application state, and `tmp/` for installer staging under the dedicated installation root. No database is initialized by the installer. Staging files are retained for inspection rather than automatically removed. Do not rerun environment installation against an active service.

The installer's environment variables apply only to its own process. After completing installation, `scripts/demo_run.sh` runs commands with the dedicated environment, workspace, cache and tools paths. It requires the source to be in `src/` under the installation root and refuses root execution. Do not start the application with an unrelated working directory or its default workspace by accident.

A transferred source snapshot can be used without a Git executable on the target. It is not a Git checkout. Keep the authoritative development branch in its original worktree; record the base commit and changed-file checksums alongside each transferred snapshot. Do not transfer `.git`, local environments, active workspaces, `.env` files or SSH keys. Preserve previous source snapshots until their replacement has been verified; never overlay an active runtime.

## Approved private runtime preparation

Run these only after approval to download software, models and public sample data. The prefetch commands validate the branch's pinned checksums, preserve incomplete downloads, and refuse to overwrite an existing checksum mismatch. They do not create projects or invoke inference. Do not run two prefetch processes for the same asset.

```bash
bash scripts/demo_run.sh python scripts/demo_prefetch.py models
bash scripts/demo_run.sh python scripts/demo_prefetch.py datasets Task09_Spleen Task05_Prostate tnbc-nuclei openslide-cmu-small hyperkvasir-tool-tracking kvasir-instrument Task06_Lung
bash scripts/demo_run.sh python scripts/demo_prefetch.py viewer slicer
bash scripts/demo_run.sh python scripts/demo_prefetch.py viewer qupath
bash scripts/demo_run.sh monailabel viewer ohif
bash scripts/demo_run.sh monailabel viewer cvat
```

Dataset identifiers must match `packages/server/src/monailabel/server/resources/dataset-templates.json` if the branch changes. Downloads remain separate from workspace imports. Model and sample licenses still apply; this is research workflow testing, not clinical validation.

nnInteractive additionally needs its isolated worker environment. To prepare it without running inference, retain the same cache and interpreter that its worker uses:

```bash
bash scripts/demo_run.sh env UV_CACHE_DIR=/opt/monailabel-demo/cache/monailabel/models/nninteractive/runtime UV_LINK_MODE=copy python -m uv sync --no-config --locked --python /opt/monailabel-demo/env/bin/python --script packages/nninteractive/src/monailabel/nninteractive/worker.py
```

The private launcher selects physical GPU 0 for annotation, limits browser desktops to two, and removes the standard OpenAI, Anthropic, Gemini and NVIDIA inference credential environment variables. This is a precaution, not a security sandbox: a trusted administrator can still configure connections in the application. Do not add paid connections during a local-only test.

`deploy/demo/monailabel-private.service` is an optional systemd **user** service template for the example `/opt/monailabel-demo` layout. Review and adapt paths before installing it. It binds the backend to loopback, runs local Lightning on physical GPU 1, and sets the browser-desktop callback to loopback. The template requires two GPUs; it is not a single-GPU default. An administrator must have separately enabled user-service lingering for operation after logout. Installing a service does not expose it to the LAN or internet; browser access from another machine needs an approved private tunnel or later networking work.

For unreliable registry DNS, `scripts/demo_prepare_coordinator.py` can provision Lightning from a previously imported Docker archive. Supply the original pinned index, AMD64 manifest, archive and imported image reference. Without `--apply` it validates only. It verifies the manifest chain, original image configuration, loaded filesystem layer digests and Git/LFS checksums in the pinned model snapshot. With `--apply` it starts the standard authenticated, loopback-only coordinator using the verified local image ID, retaining the normal lifecycle configuration label. It does not change host DNS or print the runtime key. Obtain the archive and raw manifests through an approved download route; do not replace a digest pin with an unverified tag.

FFmpeg must support the video pipeline's `-fps_mode` option. If the operating system's older FFmpeg lacks it, place a separately verified compatible build in the installation's `tools/bin`; do not replace system binaries. Source snapshots are not a single all-in-one container: the backend uses the locked Python environment, while the coordinator, CVAT and browser desktops use Docker.

## Deferred public deployment work

Docker Engine and NVIDIA Container Toolkit installation, daemon configuration, service startup and container permissions require their own reviewed step. The upstream `setup.sh` can add the invoking user to the Docker group and start/restart Docker, so do not use it to bypass that decision. Docker documents that [membership in its group grants root-level privileges](https://docs.docker.com/engine/install/linux-postinstall/).

OHIF, browser-hosted Slicer, QuPath and CVAT are supported private workflow candidates. The environment installer alone provisions none of them; use the separately approved runtime preparation above. Browser desktops use a networkless container with a session-authenticated Unix gateway, as described in [viewers](viewers.md#browser-desktop-server-setup). Their native scripting environment still requires bounded writable storage and further resource/expiry testing before public exposure. A dedicated Linux account alone is not a network or container isolation boundary.

Admission limits, one-hour visitor accounts, cleanup, TLS, routing and firewall policy remain separate approval gates. Do not publish an application merely because a private test succeeds. Use the [private test drive](demo-test-drive.md) before designing visitor restrictions.

## Offline preparation checks

Syntax and guard tests do not install anything, access the network or start application services. The standalone tests use temporary directories and block privileged/package commands:

```bash
bash -n scripts/demo_install.sh
python3 -B tests/test_demo_install.py
```

These checks establish the no-install defaults and argument guards, not successful installation or public-demo isolation. Actual dependency installation and application/viewer tests need their own approved disposable environment.
