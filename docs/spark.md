# DGX Spark

DGX Spark is a target for the Linux server. ARM64 dependency resolution and setup are implemented experimentally; the complete deployment has **not been validated on Spark hardware**. Managed Slicer/QuPath desktops and the pinned CVAT images still require an x86_64 host. OHIF is the browser viewer available for the initial Spark deployment.

Spark combines an ARM64 CPU with a GB10 GPU. Use its supported driver and NVIDIA Container Toolkit. The uv workspace selects CUDA 13 PyTorch/Torchvision wheels on Linux ARM64; a plain PyPI installation may select different builds. Native CUDA extensions must support GB10's compute capability. See the [NVIDIA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html) and [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2100).

```bash
./setup.sh
./setup.sh --check
uv run python examples/check_gpu.py
uv run monailabel-server --assistant-variant 4b
```

Setup selects the ARM64 Node archive, installs the Python workspace and prepares OHIF. It leaves unsupported desktop and CVAT images uninstalled. Nano 4B uses a pinned NVIDIA vLLM image with an ARM64 manifest; NVIDIA's [model deployment instructions](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16#use-it-with-vllm) identify Spark as a deployment target. This manifest check does not establish that MONAI Label's full runtime has passed a Spark test. Lightning's Spark runtime still needs verification.

The GPU check runs a small tensor operation, a U-Net forward/backward pass and a MONAI import in the installed environment. It does not download weights or call a hosted model. After it passes, verify local chat, VISTA3D inference/fine-tuning, U-Net training and SAM2 on representative data before relying on the deployment. A workstation test cannot replace these device checks.

Use the server's hostname or IP from another device; no software beyond a browser is needed on the client. Use HTTPS for microphone and clipboard browser permissions. ARM64 desktop viewer builds and CVAT provisioning remain in the [roadmap](roadmap.md); physical tablet interaction also needs device validation.
