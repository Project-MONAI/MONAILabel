# DGX Spark

MONAI Label runs its Linux ARM64 server and local annotation/training workloads on DGX Spark. The checks below distinguish real GPU execution and native viewers from deterministic annotation fixtures; passing a software workflow does not establish clinical accuracy.

Spark combines an ARM64 CPU with a GB10 GPU. Use its supported driver and NVIDIA Container Toolkit. The uv workspace selects CUDA 13 PyTorch/Torchvision wheels on Linux ARM64; a plain PyPI installation may select different builds. Native CUDA extensions must support GB10's compute capability. See the [NVIDIA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html) and [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2100).

```bash
./setup.sh
./setup.sh --check
uv run python examples/check_gpu.py
uv run monailabel-server --assistant-variant lightning
```

Setup selects the ARM64 Node archive, installs the Python workspace, builds OHIF and prepares native QuPath and CVAT using Docker. First setup downloads pinned sources and dependencies. QuPath builds in an isolated native JDK container. CVAT builds its server from a pinned upstream commit and serves the pinned browser assets with native Nginx; it does not run the upstream x86 server under emulation. Viewer build logs and reusable downloads stay in the tool cache.

The GPU check runs a tensor operation, a U-Net forward/backward pass and a MONAI import. It does not download weights or call a hosted model. The pinned PyTorch wheel can report that GB10's compute capability exceeds its listed build targets; keep that warning visible and verify actual kernels with the smoke checks rather than treating an import as sufficient evidence.

## Single-GPU memory

Spark shares memory between its CPU and GPU. Run one conversation runtime at a time. Managed Lightning uses a 30% serving fraction, bounded context/cache and at most four concurrent requests, leaving room for viewers, VISTA3D, SAM and U-Net. Nano 4B uses 15%; it is an experimental alternative, not an additional service to start alongside Lightning. The fraction is not a process-wide memory cap. Long concurrent conversations and annotation jobs can increase latency; wait for each Quickstart step to finish.

Use `--assistant-gpu-memory-utilization` only when an explicit override is needed. Existing external services retain their own settings; changing MONAI Label's arguments cannot rebalance another service. See [coordinator setup](coordinator.md) for cached runtime replacement and endpoint selection. No tensor parallelism across multiple GPUs is requested.

## Native Slicer

The pinned Slicer release has no managed Linux ARM64 binary recipe. Build a native portable installation explicitly, or supply an existing compatible one:

```bash
uv run python examples/build_slicer_arm64.py
uv run monailabel-server --assistant-variant lightning
```

The default tool cache is discovered automatically after the build. For a custom `--cache-dir`, use the `MONAILABEL_SLICER_EXECUTABLE` export printed by the builder. Clicking Slicer does not implicitly start a source compilation.

The optional source build uses an isolated Ubuntu 24.04 Docker builder, pinned Slicer and DCMQI revisions, a native CTK launcher and native DICOM utilities. Allow hours for first compilation and ample disk space. Intermediate objects remain cached so rerunning resumes compilation. The application uses Ubuntu's Qt 5 runtime, supplied by the ARM64 browser-desktop container and by `setup.sh` for local launches; it is not a self-contained binary for arbitrary Linux distributions. The builder does not modify system libraries or an existing Slicer installation. Extension Manager and application self-updating are disabled for this custom ARM64 build; arbitrary third-party Slicer extensions need separate ARM64 builds and testing.

## Remote access

The server listens on `127.0.0.1` by default. For LAN access, explicitly choose an interface and port:

```bash
uv run monailabel-server --host 0.0.0.0 --port 8000 --assistant-variant lightning
```

Create the first administrator from localhost, then open the server's hostname or IP from another device. Slicer and QuPath use private browser desktops for a non-loopback URL; OHIF and CVAT open their browser editors. Use HTTPS for microphone and clipboard browser permissions, and configure allowed hosts and a firewall as described in [viewer setup](viewers.md#phones-tablets-and-other-computers).

Browser tests accept `MONAILABEL_E2E_HOST` for an actual local IPv4 address. They bootstrap on loopback, then exercise the native network interface, browser-desktop selection and viewer operations using that address. Desktop checks also cover binding only that interface after bootstrap, without a loopback listener. This validates LAN URL handling on Spark, not a separate physical tablet or Safari device.

## Verification coverage

| Area | Hardware/software check | Boundary |
| --- | --- | --- |
| Radiology | Real Decathlon Spleen import, VISTA3D inference, fine-tuning, checkpoint continuation and held-out base/derived comparison on GB10 | Golden real-data runner uses deterministic conversation routing unless a coordinator is selected |
| OHIF | Native ARM-built browser distribution; volume/slice editing and submission through the LAN IP | Synthetic CT and deterministic VISTA3D/Astra annotation fixtures |
| 3D Slicer | Native ARM source build; LAN volume/slice annotation, clear/undo/redo, repair, submission/review, clipboard, resizing, draft preservation and application-exit lifecycle; native DICOM utilities | Synthetic asymmetric volume and deterministic VISTA3D/Astra annotation fixtures; no third-party extension or clinical DICOM conformance claim |
| Pathology / QuPath | Native ARM desktop; public OpenSlide sample, drawn region, submission, accepted-region U-Net training and checkpoint reuse through the LAN IP | Hosted vision response is a fixture; U-Net and the native adapter execute normally |
| Endoscopy / CVAT | Native ARM services; draft editing, clear/undo, submission, separate review and restart persistence; real SAM tracking on a HyperKvasir sample | Hosted tool localization/segmentation responses are fixtures; source-frame extraction and SAM execute normally |
| SAM 2.1 | Real CUDA image masks and 70-frame video propagation across the chunk boundary | Synthetic objects verify geometry and execution, not clinical tracking quality |
| Coordinator | Fresh managed 30B and 4B startup with their Spark budgets; real tool selection, radiology/pathology learning stories and LAN CVAT interaction | Annotation fixtures avoid paid hosted calls; one passing workflow is not a reliability or model-quality benchmark |

Run the reproducible checks in [Verification](testing.md), including `examples/check_gpu.py`, `examples/vista3d_smoke.py`, `examples/sam2_smoke.py --frames 70`, the golden workflow runners and opt-in browser suites. Standard unit tests deliberately skip native browser/GPU integration where an explicit opt-in is required. No paid hosted model was needed for these checks; validate your own provider credentials and annotation quality separately. Nano 9B, physical mobile devices and GPU-accelerated desktop rendering remain outside this verification scope.
