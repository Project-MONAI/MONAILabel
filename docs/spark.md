# DGX Spark

Use the standard [pip installation](installation.md#python-package) or run from source:

```bash
./setup.sh
./setup.sh --check
uv run python examples/check_gpu.py
uv run monailabel --assistant-variant lightning
```

Setup prepares native QuPath, CVAT and OHIF. The model check runs tensor operations and a U-Net forward/backward pass. If PyTorch warns about GB10's compute capability, verify actual execution with this check and the [model execution checks](testing.md#native-viewers).

## Memory settings

Run one conversation runtime at a time. Managed serving fractions on ARM64 are 30% for Lightning, 15% for Nano 4B and 25% for Nano 9B; these do not cap total memory. Leave room for annotation, training and viewers. The Nano variants are experimental.

Use `--assistant-gpu-memory-utilization` to override the fraction. Existing external services keep their own settings. See [coordinator setup](coordinator.md#setup) before replacing a cached runtime.

## Native Slicer

The pinned Slicer release has no managed ARM64 binary. Build it explicitly:

```bash
uv run python examples/build_slicer_arm64.py
uv run monailabel --assistant-variant lightning
```

The default cache is detected automatically. For a custom `--cache-dir`, use the `MONAILABEL_SLICER_EXECUTABLE` export printed by the builder, or point that variable at an existing compatible installation.

Allow several hours and ample disk space for the first build. Cached build files allow resuming. Local launch needs the Ubuntu Qt 5 libraries installed by `setup.sh`; browser desktops supply them in the container. Extension Manager and self-updates are disabled; additional Slicer extensions need separate ARM64 builds.

## Remote access

Create the administrator from localhost. For LAN access, start the server with:

```bash
uv run monailabel --host 0.0.0.0 --port 8000 --assistant-variant lightning
```

Use the server's hostname or IP from other devices. Configure [HTTPS, allowed hosts and firewall access](viewers.md#phones-tablets-and-other-computers).

## Verification coverage

Run the [model, browser and viewer checks](testing.md) on Spark. Set `MONAILABEL_E2E_HOST` to Spark's local IPv4 address for LAN checks. Tests on another machine do not verify GB10; physical tablet/browser behavior needs device testing.
