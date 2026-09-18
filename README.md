# MONAI Label

Annotate medical images and videos with an assistant in **[3D Slicer](https://www.slicer.org/), [QuPath](https://qupath.github.io/), [OHIF](https://ohif.org/) or [CVAT](https://www.cvat.ai/)**. Import datasets, review annotations, fine-tune models and compare their results from one web workspace.

| Overview | Datasets |
| --- | --- |
| [<img src="docs/assets/overview.png" alt="Project overview with synthetic sample and review counts" height="200">](docs/assets/overview.png) | [<img src="docs/assets/datasets.png" alt="Dataset workspace with annotation and evaluation images" height="200">](docs/assets/datasets.png) |
| **3D Slicer** | **OHIF** |
| [<img src="docs/assets/slicer.png" alt="Liver and spleen annotations with the MONAI Label review assistant in 3D Slicer" height="200">](docs/assets/slicer.png) | [<img src="docs/assets/ohif.png" alt="CT image and MONAI Label review controls in OHIF" height="200">](docs/assets/ohif.png) |
| **QuPath** | **CVAT** |
| [<img src="docs/assets/qupath.png" alt="Editable nuclei annotations and the MONAI Label assistant in QuPath" height="200">](docs/assets/qupath.png) | [<img src="docs/assets/cvat.png" alt="Editable snare polygon tracks and the CVAT assistant" height="200">](docs/assets/cvat.png) |
| **Training** | **Evaluation** |
| [<img src="docs/assets/training.png" alt="Training settings with a fixed evaluation set" height="200">](docs/assets/training.png) | [<img src="docs/assets/comparison.png" alt="Dice comparison and evaluation logs" height="200">](docs/assets/comparison.png) |

## System requirements

- Linux for local GPU workflows.
- [Python 3.12+](https://www.python.org/downloads/), [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Git](https://git-scm.com/downloads/).
- NVIDIA GPU with a compatible driver for local inference and training.
- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA GPU support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for [local chat](docs/coordinator.md#setup). CVAT also uses Docker Compose and [FFmpeg](https://ffmpeg.org/).
- Internet access and disk space for downloads and project data.
- Modern browser; [Slicer](https://download.slicer.org/) or [QuPath](https://qupath.github.io/) for their respective viewers.
- [Node.js 22](https://nodejs.org/en/download) and [Corepack](https://github.com/nodejs/corepack) to build OHIF from source.

## Quickstart

On Ubuntu/Debian, run `./setup.sh` once to install missing dependencies and prepare the viewers. It uses sudo for system packages and requires a working NVIDIA driver. Use `./setup.sh --check` to check prerequisites.

From the repository root, choose a Nemotron chat variant (Lightning is the default; 9B and 4B are experimental):

```bash
uv run monailabel-server --assistant-variant lightning
uv run monailabel-server --assistant-variant 9b
uv run monailabel-server --assistant-variant 4b
```

Or enable [NVIDIA-hosted](https://inference.nvidia.com) GPT annotation (Sol/Astra) when starting:

```bash
NV_INFERENCE_API_KEY="<your-api-key>" uv run monailabel-server
```

Open **http://localhost:8000**, create an administrator account, and wait for **Assistant ready**. Send prompts one at a time and wait for each job to finish. Inspect and apply proposals before submitting.

After code updates, restart the server and reload the browser.

### Radiology

**Main window**

> Create a project called "Radiology".

> Import 80% of Medical Decathlon Spleen images for annotation
> and the remaining 20% with labels for evaluation.

> Annotate spleen in the first 3 images using VISTA3D and submit for review.

> Open the first image in OHIF.

**OHIF assistant**

> Segment the spleen in the whole volume using VISTA3D.

> Submit this annotation for review.

**Main window — after inspecting the annotations and evaluation labels**

> Mark all reviews as good for imported evaluation samples.

> Mark all pending reviews as good.

> Create a VISTA3D model named "VISTA3D-Spleen" for spleen.

> Fine-tune VISTA3D-Spleen using the fixed Decathlon Spleen evaluation set.

> Compare VISTA3D-Spleen with VISTA3D on that same evaluation set.

### Pathology

**Main window**

> Create a project called "Pathology".

> Import the OpenSlide pathology sample.

> Open this sample in QuPath.

**QuPath assistant — draw a region first**

> Segment nuclei in the selected region using GPT-5.6 Sol.

> Submit this annotation for review.

### Endoscopy

**Main window**

> Create a project called "Endoscopy".

> Import the HyperKvasir tool-tracking sample.

> Open the video in CVAT.

**CVAT assistant**

> Use GPT-5.6 Sol to locate the snare on this frame.

> Use GPT-5.6 Sol to segment the snare on this frame.

> Use GPT-5.6 Sol to segment the snare and track it for 16 frames.

> Use GPT-5.6 Sol to segment the snare and track the whole video.

## Workspace

Projects, accounts, annotations and models are saved in the Git-ignored `workspace/` folder. Downloads and managed tools are cached in `workspace/.cache/`. To use a different location:

```bash
uv run monailabel-server --data-dir /path/to/workspace
```

To start fresh, stop the server and viewers, delete the workspace contents (including `.cache/`) and restart. This removes all local accounts and project data.

This is a development preview. See [current limitations](docs/roadmap.md).
