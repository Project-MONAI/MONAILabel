# Demo credits

[Watch the MONAI Label 1.0 demo](https://github.com/user-attachments/assets/9cd51524-68a5-4d12-a79d-11fd49ed3cb9) (10:05).

The video shows real application recordings and model inference on public samples. Narration is synthetic, using Microsoft Edge's en-US-GuyNeural voice. Loading intervals are trimmed and the Spark result inspection is slowed for narration. Segmentation overlays, windowing and navigation are adaptations.

| Footage | Source and terms |
| --- | --- |
| CT spleen | Medical Segmentation Decathlon Task09, Memorial Sloan Kettering Cancer Center; cases 10, 12, 13, 14 and 16. [Dataset](http://medicaldecathlon.com/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), [Antonelli et al.](https://doi.org/10.1038/s41467-022-30695-9). |
| MRI prostate import | Medical Segmentation Decathlon Task05, Radboud University Nijmegen Medical Centre; all 32 labeled T2 cases. [Dataset](http://medicaldecathlon.com/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), [Antonelli et al.](https://doi.org/10.1038/s41467-022-30695-9). |
| Pathology | `CMU-1-Small-Region.svs`, [OpenSlide Aperio test data](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/), CC0; [Goode et al.](https://doi.org/10.4103/2153-3539.119005). |
| Endoscopy | HyperKvasir snare clip `99b387e7-d07b-4268-9226-4df450c2a198`, Borgli et al. [Source record](https://osf.io/mh9sj/), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), [paper](https://doi.org/10.1038/s41597-020-00622-y). |

Retain these credits and the source terms when sharing. The Decathlon image adaptations retain CC BY-SA 4.0; the HyperKvasir excerpt and adaptations retain CC BY-NC 4.0. The complete video is not offered for unrestricted commercial reuse. These media terms do not change MONAI Label's Apache-2.0 code license.

The model runs use [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) ([paper](https://arxiv.org/abs/2503.08373)), [SAM 2.1](https://github.com/facebookresearch/sam2) ([paper](https://arxiv.org/abs/2408.00714)), [VISTA3D](https://huggingface.co/MONAI/vista3d) ([paper](https://openaccess.thecvf.com/content/CVPR2025/html/He_VISTA3D_A_Unified_Segmentation_Foundation_Model_For_3D_Medical_Imaging_CVPR_2025_paper.html)), [TotalSegmentator CT](https://github.com/wasserth/TotalSegmentator) ([paper](https://doi.org/10.1148/ryai.230024)) and [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) ([paper](https://doi.org/10.1038/s41592-020-01008-z), [ResEnc reference](https://arxiv.org/abs/2404.09556)). Model weights are not bundled with the video. nnInteractive weights have CC BY-NC-SA 4.0 terms; VISTA3D weights have separate noncommercial research/evaluation terms. SAM 2.1 and TotalSegmentator's `total` task use Apache-2.0 terms. GPT Astra runs through a configured NVIDIA gateway and remains subject to the hosted service terms. See [third-party notices](../THIRD_PARTY_NOTICES.md) for model and viewer licenses.

The Spark section uses a tablet browser emulation, not a physical iPad or microphone recording. The eight-update nnU-Net run demonstrates execution, not useful accuracy; CT/MRI examples and the pathology/endoscopy U-Net examples show setup. VISTA3D's training sources include Decathlon, so the comparison is not an independent benchmark.
