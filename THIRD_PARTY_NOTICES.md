# Third-party notices

MONAI Label's own code is licensed under Apache-2.0; see [LICENSE](LICENSE). Third-party code, data, model weights and applications retain their own licenses. Installing or downloading them does not relicense them under MONAI Label's license.

## Bundled source and resources

| Component | Source and revision | License and retained notices | Changes |
| --- | --- | --- | --- |
| MedSAM2 / SAM 2 / EfficientTAM runtime | [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383), revision `332f30d420f1d1b08e2a79b3ae6a602458808383` | Apache-2.0; `packages/sam-runtime/LICENSE`, `NOTICE`, source copyright headers and `upstream.json` | Python and YAML sources copied unchanged from `sam2` and `efficient_track_anything`. Model weights, upstream application/training directories and CUDA binaries/extensions excluded. MONAI Label supplies distribution metadata. |
| SAM model definitions | [MedSAM2](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383) and [facebookresearch/sam2](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4) | Apache-2.0; `packages/sam/src/monailabel/sam/resources/LICENSE-SAM` and accompanying README | Selected upstream YAML definitions copied unchanged under local filenames. |
| Anatomical color table | [3D Slicer](https://github.com/Slicer/Slicer/blob/v5.12.0/Base/Logic/Resources/ColorFiles/GenericAnatomyColors.txt), distributed with Slicer 5.12.4 | Slicer Contribution and Software License Agreement; `packages/core/src/monailabel/core/resources/GenericAnatomyColors.LICENSE.txt`, original table header and resource README | Color table copied unchanged; MONAI Label uses names and colors independently of upstream numeric IDs. |
| VISTA3D target vocabulary | [MONAI/vista3d](https://huggingface.co/MONAI/vista3d/tree/c6dbe159632a4767696e09f91d74d729b82e73e6), bundle 0.5.11 | Upstream code license: Apache-2.0; `packages/providers/src/monailabel/providers/resources/LICENSE-VISTA3D` | `vista3d_labels.json` selects the bundle's 117 default automatic targets plus the explicitly prompted lung tumor class (23). Model weights have separate terms in the upstream license and are downloaded separately. |
| Chat Markdown renderer | [markdown-it 15.0.2](https://github.com/markdown-it/markdown-it/tree/15.0.2) | MIT; bundled dependencies use MIT and BSD-2-Clause | The unchanged browser bundle, component versions, hash and all license texts are retained in `packages/server/src/monailabel/server/static/vendor/`. |
| Nemotron 9B tool parser | [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/6533e8de2c68e4536bf7c411d7a3ce5734111476/nemotron_toolcall_parser_no_streaming.py), revision `6533e8de2c68e4536bf7c411d7a3ce5734111476` | Apache-2.0 SPDX header; MONAI Label's Apache license is included in the server distribution | Adapted for vLLM 0.17 public imports and non-streaming transport; source and modification notice retained at the top of `nemotron9_parser.py.txt`. |
| MONAI icon on the VISTA3D card | [Project-MONAI/MONAI favicon](https://github.com/Project-MONAI/MONAI/blob/cf3e11bb90de44b5bd02a8887e325fc6f629350e/docs/images/favicon.ico) | Copyright MONAI Consortium; Apache-2.0, retained in `packages/server/src/monailabel/server/static/model-icons/LICENSE-MONAI.txt` | Copied unchanged; used to identify the model's origin. Source and hash are recorded in the accompanying README. Trademark rights remain with MONAI. |

The release checks verify runtime snapshot hashes and required license files. Vendored source is excluded from automatic formatting so upstream copyright headers and file contents remain intact.

## Downloaded model weights

Weights are downloaded separately; code licenses do not determine weight licenses. These terms also matter when distributing derived checkpoints.

| Model | Weight terms | Citation |
| --- | --- | --- |
| VISTA3D 0.5.11 | [NVIDIA license in the pinned bundle](https://huggingface.co/MONAI/vista3d/blob/c6dbe159632a4767696e09f91d74d729b82e73e6/LICENSE): noncommercial research/evaluation only, including derived weights. The full text is retained in `LICENSE-VISTA3D`. | [He et al., VISTA3D, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/He_VISTA3D_A_Unified_Segmentation_Foundation_Model_For_3D_Medical_Imaging_CVPR_2025_paper.html) |
| SAM 2.1 | [Apache-2.0, including checkpoints](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4#license) | [Ravi et al., SAM 2](https://arxiv.org/abs/2408.00714) |
| MedSAM2 | The [pinned model card](https://huggingface.co/wanglab/MedSAM2/blob/e4a6f35edd7e091619cbc0750f462f1574e23955/README.md) declares CC BY-SA 4.0 in metadata but explicitly restricts weights to research and education. Treat that restriction as applicable; obtain clarification from the authors before commercial use or redistribution under broader terms. Its runtime code is separately Apache-2.0. | [Ma et al., MedSAM2](https://arxiv.org/abs/2504.03600) |
| nnInteractive v1.0 | [Pinned model terms](https://huggingface.co/MIC-DKFZ/nnInteractive/blob/3f308d751c00644e4fde6f09c600264b393b21b5/nnInteractive_v1.0/LICENSE): CC BY-NC-SA 4.0; noncommercial use, attribution and share-alike. Downloaded separately; not bundled in wheels or images. | [Isensee et al., nnInteractive](https://arxiv.org/abs/2503.08373) |
| Nemotron Nano 4B / 9B | [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/); retain each pinned model card's notices, including “Improved using Qwen” where present. | [4B model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/README.md), [9B model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/6533e8de2c68e4536bf7c411d7a3ce5734111476/README.md) |

Hosted models remain subject to their provider's service and model terms. The configured local Lightning model's terms are linked in its [pinned model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/blob/cc84af2fe71647d87f4486c064f320e1e7535243/README.md); the serving image has separate terms.

## TotalSegmentator models

The CT `total` and MRI `total_mr` tasks use [TotalSegmentator 2.18.0](https://github.com/wasserth/TotalSegmentator/tree/v2.18.0), Apache-2.0. The adapter includes the upstream class mapping, converted to JSON, and retains `packages/totalsegmentator/src/monailabel/totalsegmentator/resources/LICENSE-TotalSegmentator`. Checkpoint URLs, archive SHA-256 values and per-file hashes are recorded in the accompanying `weights.json`. The 3 mm CT checkpoint is from `v2.0.0-weights`; the 3 mm MRI checkpoint is from `v2.5.0-weights`. Weights are downloaded separately and are not bundled in wheels or images. Other upstream tasks can have different terms and are not exposed here.

Cite Wasserthal et al., *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images*, Radiology: Artificial Intelligence (2023), [DOI 10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024); for MRI also cite Akinci D’Antonoli et al., *TotalSegmentator MRI: Robust Sequence-independent Segmentation of Multiple Anatomic Structures in MRI*, Radiology (2025), [DOI 10.1148/radiol.241613](https://doi.org/10.1148/radiol.241613). The runtime uses [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), Apache-2.0; cite Isensee et al., [DOI 10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z).

The Prostate MRI template derives whole-prostate masks by combining Decathlon Task05 PZ and TZ labels on their original grid. The original and derived masks retain CC BY-SA 4.0; the template credits Medical Segmentation Decathlon and Radboud University Nijmegen Medical Centre.

The Decathlon Lung template names Task06's published `cancer` label `lung tumor`; segmentation coverage is unchanged and retains CC BY-SA 4.0. The dataset source and license are recorded with each import.

TotalSegmentator CT/MRI dataset templates use CC BY 4.0; the pinned source record, dataset DOI and attribution are displayed in the importer. These collections include upstream model development data.

## nnU-Net v2

The training recipe uses [nnU-Net 2.6.4](https://github.com/MIC-DKFZ/nnUNet/tree/v2.6.4), Apache-2.0, through MONAI's planning/preprocessing wrapper and upstream training/inference APIs. It is installed as a separate distribution with its license retained in `nnunetv2-2.6.4.dist-info/licenses/LICENSE`. No upstream weights are bundled. Cite [Isensee et al., nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z) when using nnU-Net. Datasets and imported weights retain their own terms.

The ResEnc L preset also requests citation of [Isensee et al., nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation](https://arxiv.org/abs/2404.09556).

## nnInteractive

[nnInteractive 2.6.0](https://github.com/MIC-DKFZ/nnInteractive) and nnU-Net 2.8.1 are installed in an isolated inference environment, retaining their Apache-2.0 license files. Dependency versions are pinned in `packages/nninteractive/src/monailabel/nninteractive/worker.py.lock`; checkpoint sizes and SHA-256 hashes are pinned in `weights.py`. The downloaded checkpoint's CC BY-NC-SA 4.0 terms are separate from the code license.

[SlicerNNInteractive](https://github.com/coendevente/SlicerNNInteractive) and [OHIF-AI](https://github.com/CCI-Bonn/OHIF-AI) were consulted for interaction and coordinate conventions. Their code and assets are not bundled; MONAI Label implements its own viewer controls and backend adapter.

## Installed dependencies and containers

Python dependencies are separate distributions. Their license metadata and license files remain in their installed `.dist-info` directories. The server image includes `/usr/share/monailabel/pip-inspect.json`, recording installed versions and package metadata, plus this notice inventory. This inventory does not replace each dependency's license text.

The Docker image retains Debian package copyright files under `/usr/share/doc`, Python's standard-library license, the Node.js license under `/usr/share/doc/node/LICENSE`, and npm/Corepack licenses under `/usr/local/lib/node_modules`. Docker CLI and Compose are installed from Docker's signed Debian repository with their package notices. Published image builds also generate an SBOM and build provenance.

Managed viewers and serving runtimes are downloaded separately from their upstream projects: [Slicer](https://github.com/Slicer/Slicer/blob/v5.12.0/License.txt), [QuPath](https://github.com/qupath/qupath/blob/main/LICENSE), [OHIF](https://github.com/OHIF/Viewers/blob/master/LICENSE), [CVAT](https://github.com/cvat-ai/cvat/blob/develop/LICENSE), [noVNC](https://github.com/novnc/noVNC/blob/master/LICENSE.txt), and the serving images identified in `providers/chat/local_models.py`. Their distributions retain upstream licenses. Model licenses and usage terms apply to downloaded weights independently of runtime code.

## Dataset templates

Dataset files are downloaded from the cited publisher into the user's cache; they are not bundled in MONAI Label wheels or server images. Each catalog entry displays its source and dataset-specific license/usage terms. Preserve those terms and the cited dataset paper when exporting or redistributing data or reporting results. Some collections restrict use to research/education or noncommercial purposes; MONAI Label's Apache license does not remove those restrictions.

For Medical Segmentation Decathlon, cite Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications (2022), [DOI 10.1038/s41467-022-30695-9](https://doi.org/10.1038/s41467-022-30695-9), and retain the source institution attribution supplied with each task. HyperKvasir uses Borgli et al., Scientific Data (2020), [DOI 10.1038/s41597-020-00622-y](https://doi.org/10.1038/s41597-020-00622-y). OpenSlide uses Goode et al., J Pathol Inform (2013), [DOI 10.4103/2153-3539.119005](https://doi.org/10.4103/2153-3539.119005); its sample image terms are independent of the viewer's code license.

The TNBC nuclei template uses the [Naylor et al. release](https://zenodo.org/records/1175282), licensed CC BY 4.0. Attribute the authors and dataset DOI. The [Kvasir-Instrument template](https://datasets.simula.no/kvasir-instrument/) uses the publisher-linked OSF archives: its terms restrict use to research/education, require prior written permission for commercial use and require citation of Jha et al., MMM 2021, DOI `10.1007/978-3-030-67835-7_19`. These terms are shown in the importer and recorded with the import job. See the repository's `docs/datasets.md` for the implemented collections and grouping rules.

## Documentation images

Repository screenshots include Decathlon CT, OpenSlide pathology and HyperKvasir video images. Source attribution, modifications and dataset-specific image licenses are recorded in [screenshot credits](docs/assets/README.md), including the corresponding frames of `gallery.gif`. These images are not covered solely by the project’s Apache code license.

## Updating third-party material

For a source update, record the exact upstream revision, retain the upstream license and any NOTICE/copyright files, describe modifications, refresh the provenance hashes and rerun distribution checks. Keep model/data licensing separate from the code license. The installed-dependency inventory is a record of the shipped environment, not a blanket assertion that every possible use is permitted.
