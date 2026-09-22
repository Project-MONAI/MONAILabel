# MONAI Label

MONAI Label is an open-source image labeling and learning tool for interactive AI annotation of medical images and videos. Use an assistant in **[3D Slicer](https://www.slicer.org/), [QuPath](https://qupath.github.io/), [OHIF](https://ohif.org/) or [CVAT](https://www.cvat.ai/)** for radiology, pathology and endoscopy.

Import datasets, review annotations, train and fine-tune models, and compare their results from one web workspace.

<p align="center">
<picture>
<source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/overview.png">
<img src="docs/assets/gallery.gif" alt="MONAI Label workspace and viewers" width="100%">
</picture>
</p>

<p align="center">
<a href="docs/assets/overview.png"><img src="docs/assets/overview.png" alt="Overview" title="Overview" width="7%"></a>&nbsp;
<a href="docs/assets/datasets.png"><img src="docs/assets/datasets.png" alt="Datasets" title="Datasets" width="7%"></a>&nbsp;
<a href="docs/assets/models.png"><img src="docs/assets/models.png" alt="Models" title="Models" width="7%"></a>&nbsp;
<a href="docs/assets/reviews.png"><img src="docs/assets/reviews.png" alt="Reviews" title="Reviews" width="7%"></a>&nbsp;
<a href="docs/assets/slicer.png"><img src="docs/assets/slicer.png" alt="3D Slicer" title="3D Slicer" width="7%"></a>&nbsp;
<a href="docs/assets/ohif.png"><img src="docs/assets/ohif.png" alt="OHIF" title="OHIF" width="7%"></a>&nbsp;
<a href="docs/assets/qupath.png"><img src="docs/assets/qupath.png" alt="QuPath" title="QuPath" width="7%"></a>&nbsp;
<a href="docs/assets/cvat.png"><img src="docs/assets/cvat.png" alt="CVAT" title="CVAT" width="7%"></a>&nbsp;
<a href="docs/assets/training.png"><img src="docs/assets/training.png" alt="Training" title="Training" width="7%"></a>&nbsp;
<a href="docs/assets/comparison.png"><img src="docs/assets/comparison.png" alt="Evaluation" title="Evaluation" width="7%"></a>
</p>

## System requirements

- Linux for local GPU workflows.
- DGX Spark: [experimental ARM64 server setup](docs/spark.md).
- [Python 3.12+](https://www.python.org/downloads/), [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Git](https://git-scm.com/downloads/).
- NVIDIA GPU with a compatible driver for local inference and training.
- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA GPU support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for [local chat](docs/coordinator.md#setup). CVAT also uses Docker Compose and [FFmpeg](https://ffmpeg.org/).
- Internet access and disk space for downloads and project data.
- Modern browser. Slicer and QuPath launch normally on localhost; remote clients use a browser desktop hosted by the Linux server with Docker. No viewer installation is needed on remote clients.
- [Node.js 22](https://nodejs.org/en/download) and [Corepack](https://github.com/nodejs/corepack) to build OHIF from source.

## Quickstart

On Ubuntu/Debian, run `./setup.sh` once to install missing dependencies and prepare the viewers. It uses sudo for system packages and requires a working NVIDIA driver. Use `./setup.sh --check` to check prerequisites.

For hosted annotation, use an [NVIDIA Inference API key](https://inference.nvidia.com) or your provider's API key. Predefined models prefer NVIDIA; you can change their provider in **Models**.

Set `NV_INFERENCE_API_KEY` (NVIDIA), `OPENAI_API_KEY` (OpenAI), `ANTHROPIC_API_KEY` (Claude) or `GEMINI_API_KEY` (Gemini) before starting the server. **Models → Add model** lists compatible models to import and name for your project ([setup](docs/providers.md)).

Start the server. Local chat uses Nemotron 3.5 Lightning by default; Nano 9B and Nano 4B are experimental options for smaller GPUs:

```bash
export NV_INFERENCE_API_KEY="<your-api-key>"
uv run monailabel-server

# If you have a smaller GPU, try Nemotron Nano 9B or Nano 4B:
uv run monailabel-server --assistant-variant 9b
uv run monailabel-server --assistant-variant 4b
```

Open **http://localhost:8000**, create an administrator account, and wait for **Assistant ready**. From another device, use the server's hostname or IP instead of `localhost`. Send prompts one at a time and wait for each job to finish. Inspect and apply proposals before submitting.

Send each prompt below in order from the window named in the comment. Choose the specialty that matches your data.

<details open>
<summary><b>Radiology</b> — Decathlon Spleen, VISTA3D, OHIF</summary>

```text
# Main window
  Create a project called "Radiology".
  Import Decathlon Spleen: 80% to annotate, 20% with evaluation labels.
  Annotate spleen in the first 3 images with VISTA3D; submit for review.
  Open the first image in OHIF.

# OHIF assistant
  Segment the spleen in the whole volume using VISTA3D.
  Clear the spleen annotation on the current slice.
  Annotate the spleen on the current slice using GPT Astra.
  Submit this annotation for review.

# Main window, after inspecting the annotations and evaluation labels
  Mark all reviews as good for imported evaluation samples.
  Mark all pending reviews as good.
  Create a VISTA3D model named "VISTA3D-Spleen" for spleen.
  Fine-tune VISTA3D-Spleen with the Decathlon Spleen evaluation set.
  Compare VISTA3D-Spleen with VISTA3D on that same evaluation set.

# Or train a new model from the approved annotations
  Create a U-Net model named "Spleen U-Net" for spleen.
  Train Spleen U-Net with approved samples.

# OHIF assistant
  Segment the spleen using Spleen U-Net.
```

The extended, tested Spleen workflow is in [datasets, models and learning](docs/workflows.md#try-the-spleen-learning-workflow).

</details>

<details open>
<summary><b>Pathology</b> — OpenSlide sample, nuclei, QuPath</summary>

```text
# Main window
  Create a project called "Pathology".
  Import the OpenSlide pathology sample.
  Open this sample in QuPath.

# QuPath assistant, after drawing a region
  Segment nuclei in the selected region using GPT Astra.
  Clear all annotations in the selected region.
  Segment nuclei in the selected region using GPT Astra.
  Submit this annotation for review.

# Main window, after inspecting each submitted region in Reviews
  Mark all pending reviews as good.
  Create a U-Net model named "Nuclei U-Net" for nuclei.
  Train Nuclei U-Net with approved samples.

# QuPath assistant
  Segment nuclei in the selected region using Nuclei U-Net.
```

</details>

<details open>
<summary><b>Endoscopy</b> — HyperKvasir clip, tool tracking, CVAT</summary>

```text
# Main window
  Create a project called "Endoscopy".
  Import the HyperKvasir tool-tracking sample.
  Open the video in CVAT.

# CVAT assistant
  Locate the snare on this frame.
  Segment the snare and track it for 16 frames.
  Clear the snare annotations for 16 frames.
  Undo that.
  Segment the snare and track the whole video.
  Submit this annotation for review.

# Main window, after inspecting the submitted frame ranges in Reviews
  Mark all pending reviews as good.
  Create a U-Net model named "Snare U-Net" for snare.
  Train Snare U-Net with approved samples.

# CVAT assistant
  Segment the snare on this frame using Snare U-Net.
```

</details>

Training can start without an evaluation dataset. With several independent cases, try: `Train Snare U-Net with an 80:20 train/evaluation split.` Regions from one slide and frames from one procedure stay together.

## Workspace

Projects, accounts, annotations and models are saved in the Git-ignored `workspace/` folder. Downloads and managed tools are cached in `workspace/.cache/`. To use a different location:

```bash
uv run monailabel-server --data-dir /path/to/workspace
```

To start fresh, stop the server and viewers, delete the workspace contents (including `.cache/`) and restart. This removes all local accounts and project data.

This is a development preview. See the [design guide](docs/design.md) and [current limitations](docs/roadmap.md).
