# CVAT MONAILabel extension

## Requirement

Install CVAT and enable Semi-Automatic and Automatic Annotation

- https://openvinotoolkit.github.io/cvat/docs/getting_started
- https://openvinotoolkit.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/

> `docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d`

## Installation

Run `./deploy.sh` to install all available models from MONAI Label into CVAT.
Currently, following sample models are available for CVAT.

- [Segmentation Nuclei](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/pathology#pathology-use-case) ([Detector](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/ai-tools/#detectors))
- [Deepedit Nuclei](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/pathology#pathology-use-case) ([Detector](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/ai-tools/#detectors))
- [NuClick](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/pathology#pathology-use-case) ([Interactor](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/ai-tools/#interactors))

## Using Plugin

> Currently we can use MONAI Label models only for annotation. Other features like ActiveLearning, Finetuning/Training
> models is not supported.
>
> Annotation functions are verified only on basic png/jpeg images in CVAT.

### Models

![image](../../docs/images/cvat_models.jpeg)

### Detector

![image](../../docs/images/cvat_detector.jpeg)

### Interactor

> Currently CVAT supports single polygon as result for Interactor. Hence, NuClick model in CVAT will return only one
> polygon mask.

![image](../../docs/images/cvat_interactor.jpeg)

