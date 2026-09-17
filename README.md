# MONAI Label

Annotate medical images with an assistant in **3D Slicer, QuPath or OHIF**. Import datasets, review annotations, fine-tune models and compare their results from one web workspace.

## Screenshots

Click a preview to enlarge.

| Datasets and workspace chat | CT annotation and review in 3D Slicer |
| --- | --- |
| [<img src="docs/assets/datasets.png" alt="Dataset workspace with annotation and evaluation images" height="200">](docs/assets/datasets.png) | [<img src="docs/assets/slicer.png" alt="Liver and spleen annotations with the MONAI Label review assistant in 3D Slicer" height="200">](docs/assets/slicer.png) |
| **Browser review in OHIF** | **Pathology annotation in QuPath** |
| [<img src="docs/assets/ohif.png" alt="CT image and MONAI Label review controls in OHIF" height="200">](docs/assets/ohif.png) | [<img src="docs/assets/qupath.png" alt="Editable nuclei annotations and the MONAI Label assistant in QuPath" height="200">](docs/assets/qupath.png) |
| **Fine-tune a named model** | **Compare models and inspect logs** |
| [<img src="docs/assets/training.png" alt="Training settings with a fixed evaluation set" height="200">](docs/assets/training.png) | [<img src="docs/assets/comparison.png" alt="Dice comparison and evaluation logs" height="200">](docs/assets/comparison.png) |

Screenshots use public Decathlon Spleen and CMU pathology samples. Scores illustrate the workflow, not clinical performance.

## System requirements

- Linux for local GPU workflows.
- [Python 3.12+](https://www.python.org/downloads/), [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Git](https://git-scm.com/downloads/).
- NVIDIA GPU with a compatible driver for local inference and training.
- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA GPU support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for [local chat](docs/coordinator.md#setup).
- Internet access and disk space for downloads and project data.
- Modern browser; [Slicer](https://download.slicer.org/) or [QuPath](https://qupath.github.io/) for their respective viewers.
- [Node.js 22](https://nodejs.org/en/download) and [Corepack](https://github.com/nodejs/corepack) to build OHIF from source.

## Quickstart

On Ubuntu/Debian, run `./setup.sh` once to install missing dependencies and prepare the viewers. It uses sudo for system packages and requires a working NVIDIA driver. Use `./setup.sh --check` to check prerequisites.

Run from the repository root:

```bash
uv run monailabel-server
```

The command installs the required Python dependencies and starts the server. Model weights download when needed and are cached for reuse.

Open **http://localhost:8000** and create your administrator account. There is no default password. The assistant footer shows when chat is ready.

### Try the full workflow

Send these prompts **one at a time** in the web chat. Wait for each job to finish before continuing; Activity shows progress and logs. This example uses local VISTA3D and the Medical Decathlon Spleen dataset.

<!-- spleen-prompts:start -->

```text
Create new project "Radiology - Spleen Segmentation"

From Medical Decathlon Spleen Dataset import 80% images only for annotation and remaining 20% images+labels for evaluation.

Mark all reviews as good for imported evaluation samples.

Run VISTA3D segmentation to annotate spleen for first 5 images and submit for review.

Mark all pending reviews as good

Create VISTA 3D based model "VISTA3D-Spleen" for spleen

Finetune VISTA3D-Spleen and use fixed Decathlon Spleen set for evaluation

Compare VISTA3D-Spleen vs VISTA3D against fixed Decathlon Spleen set
```

<!-- spleen-prompts:end -->

The split imports 32 images for annotation and 9 image/label pairs for independent evaluation. Review predictions before accepting them for training. Training creates a derived model; comparison reports Dice scores against the same held-out references.

For a quick **CPU demo without model API keys**, choose **Explore synthetic demo** in the workspace. It uses generated images, simulated review and a simple baseline; chat still needs a conversation model.

## Workspace

Projects, accounts, annotations and models are saved in the Git-ignored `workspace/` folder. Downloads and managed tools are cached in `workspace/.cache/`. To use a different location:

```bash
uv run monailabel-server --data-dir /path/to/workspace
```

To start fresh, stop the server and viewers, delete the workspace contents (including `.cache/`) and restart. This removes all local accounts and project data.

This is a development preview. See [current limitations](docs/roadmap.md).
