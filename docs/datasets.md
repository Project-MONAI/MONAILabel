# Dataset templates

Open **Datasets → Sample datasets**. Choose images only, images with labels, or evaluation-only imports. The default imports five samples; **All samples** imports the collection. Published labels reserved for evaluation are accepted on import and ready to use. Labels imported for annotation/training still need review.

| Template | Contents | License / source |
| --- | --- | --- |
| Medical Segmentation Decathlon | [Ten CT/MRI tasks and labels](https://www.nature.com/articles/s41467-022-30695-9); labeled training and unlabeled test sections | [CC BY-SA 4.0](http://medicaldecathlon.com/dataaws/) |
| TotalSegmentator CT | 102-case sample or full archive; select up to 31 structures | [CC BY 4.0](https://zenodo.org/records/10047263) |
| Prostate MRI · Whole gland | T2/ADC MRI with PZ/TZ labels combined into one prostate mask; 240 MB | [CC BY-SA 4.0](http://medicaldecathlon.com/dataaws/) |
| TotalSegmentator MRI | 616 MRI scans with 50 structures; select up to 31; 5.1 GB | [CC BY 4.0](https://zenodo.org/records/14710732) |
| TNBC nuclei | 50 H&E patches, 11 patient groups, [binary nuclei masks](https://zenodo.org/records/1175282); 25 MB | [CC BY 4.0](https://zenodo.org/records/1175282) |
| Kvasir-Instrument | 590 endoscopy frames with [binary instrument masks](https://datasets.simula.no/kvasir-instrument/); 169 MB | [Research/education only; commercial use requires written permission](https://datasets.simula.no/kvasir-instrument/) |
| OpenSlide sample | One small pathology image without masks, for QuPath | [Source terms](https://openslide.cs.cmu.edu/download/openslide-testdata/) |
| HyperKvasir tool-tracking clip | One endoscopy video without reference tracks, for CVAT | [CC BY-NC 4.0, supplied with the OSF release](https://osf.io/mh9sj/) |

**Labels and structures** in the sample-dataset dialog links to the author's class reference. MoNuSeg, CAMELYON16 and BACH entries link to the authors' downloads; automatic import is unavailable.

Decathlon Lung imports its published `cancer` class as `lung tumor`, matching VISTA3D's target name. Mask coverage and geometry are preserved; labels retain CC BY-SA 4.0.

## Try TotalSegmentator

See the [TotalSegmentator class details](https://github.com/wasserth/TotalSegmentator#class-details) for CT/MRI structures. **Models → Model details → Supported structures** lists the classes available in MONAI Label.

For CT, select **TotalSegmentator CT · 102-case sample**, import a few images, then ask:

```text
Annotate spleen and liver using TotalSegmentator CT.
```

For MRI, select **Prostate MRI · Whole gland**, **T2**, and a few images, then ask:

```text
Annotate the prostate using TotalSegmentator MRI.
```

Choose images with labels to practice reviewing references and fine-tuning. The prostate template combines Decathlon PZ/TZ masks into a whole-gland reference on the original grid; derived masks retain CC BY-SA 4.0.

The larger **TotalSegmentator MRI · 616 cases** template offers 50 structures. TotalSegmentator CT/MRI collections include upstream development data; use a separate dataset for independent evaluation of these pretrained models or their descendants.

## Labeled pathology

TNBC imports one `nuclei` class on the original pixel grid. Quick samples visit different patients first. All patches from a patient stay in the same source group.

```text
Import TNBC nuclei: 80% to annotate, 20% with evaluation labels.
```

Percentages select whole patient groups, so image counts can differ from the requested ratio. At least two groups are required. Choose images with labels to include training references, then review them before training.

Cite Naylor et al., *Segmentation of Nuclei in Histopathology Images by Deep Regression of the Distance Map*, IEEE TMI (2019), and dataset DOI [10.5281/zenodo.1175282](https://doi.org/10.5281/zenodo.1175282).

## Labeled endoscopy and video

Kvasir-Instrument imports one `instrument` class. Its still images have no published patient/procedure mapping, so all frames share one conservative source group. Use the collection for training or for evaluation of a model trained elsewhere; automatic percentage splitting is unavailable.

Cite Jha et al., *Kvasir-Instrument: Diagnostic and Therapeutic Tool Segmentation Dataset in Gastrointestinal Endoscopy*, MMM (2021), DOI [10.1007/978-3-030-67835-7_19](https://doi.org/10.1007/978-3-030-67835-7_19). Follow the publisher's research/education and commercial-permission terms.

Use the [HyperKvasir clip](video.md#try-the-tool-tracking-sample) for video tracking. It has no supplied tracks; accepted polygon annotations can train a segmentation model.

## Downloads and attribution

A subset still requires downloading the full archive. Verified downloads are reused from `workspace/.cache/datasets/`; `MONAILABEL_DATASETS_DIR` overrides the location. Interrupted partial downloads restart. Re-importing preserves existing annotations and review decisions. Repeating an older evaluation import accepts unchanged published references that have no review decision.

Source links, terms, citations and checksums are retained in the catalog/import jobs. Follow the source license when reusing or redistributing data. Dataset files are not bundled in wheels or server images. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Export samples

Select samples in Datasets and choose **Export selected**. Choose the latest saved annotation or all saved versions, then **Prepare ZIP**. Preparation runs in the background; download the ZIP from the progress window or Activity.

The archive contains original images/videos, saved masks or tracks, and a manifest with class names, IDs and colors, source geometry, patient/slide/procedure groups, dataset use, annotation revisions and review decisions. NIfTI masks retain the source grid and affine. Region and frame-range review references retain their coverage. Unsaved viewer drafts are excluded.
