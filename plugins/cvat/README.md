# CVAT MONAILabel extension

## Requirement

Install CVAT and enable Semi-Automatic and Automatic Annotation

- https://openvinotoolkit.github.io/cvat/docs/getting_started
- https://openvinotoolkit.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/

```
# For Reference
export CVAT_HOST=127.0.0.1
docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'

wget https://github.com/nuclio/nuclio/releases/download/1.5.16/nuctl-1.5.16-linux-amd64
chmod +x nuctl-1.5.16-linux-amd64
mv nuctl-1.5.16-linux-amd64 ~/.local/bin/nuctl
```

## Installation

Run `./deploy.sh` to install all available models from MONAI Label into CVAT.
Currently, following sample models are available for CVAT.

### Endoscopy
- [ToolTracking](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/endoscopy) ([Detector](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/ai-tools/#detectors))

### Pathology
- [Segmentation Nuclei](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/pathology#pathology-use-case) ([Detector](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/ai-tools/#detectors))
- [Deepedit Nuclei](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/pathology#pathology-use-case) ([Detector](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/ai-tools/#detectors))
- [NuClick](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/pathology#pathology-use-case) ([Interactor](https://openvinotoolkit.github.io/cvat/docs/manual/advanced/ai-tools/#interactors))


If you want to deploy single model (e.g. **_tooltracking_** model for **_endoscopy_**) then you can run:
```
func_root=`pwd` # this should be plugins/cvat
func_config=endoscopy/tooltracking.yaml

nuctl create project cvat
nuctl deploy --project-name cvat --path "$func_root" --file "$func_config" --platform local
```

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

