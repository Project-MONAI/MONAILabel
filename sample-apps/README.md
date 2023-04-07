# MONAI Bundle Sample Applications

MONAI Label makes it easy to create labeling applications in a serverless way, enabling easy deployment through the MONAI Label server.

### Table of Contents
- [Using MONAI Label CLI to Pull Apps](#using-monai-label-cli-to-pull-apps)
- [Available Template Applications](#available-template-applications)
  - [Radiology](#radiology)
  - [Pathology](#pathology)
  - [Endoscopy](#endoscopy)
  - [MONAI Bundle](#monai-bundle)
- [Creating a Custom App](#creating-a-custom-app)

### Using MONAI Label CLI to Pull Apps
Currently, MONAI Label offers four template applications that can be used as-is or modified to meet specific needs. To download these templates using the MONAI Label CLI, use the following command:

```
monailabel apps --download --name <sample app name> --output <output folder>
```

### Available Template Applications

#### [Radiology](./radiology)
The radiology template includes example models for interactive and automated segmentation of radiology (3D) images. It provides examples for the following three types of models:the following three types of models:
- DeepEdit (Interactive + Auto Segmentation)
  - Spleen, Liver, Kidney, and others.
- Deepgrow (Interactive)
  - Any organ or tissue, but pre-trained to work well on Spleen, Liver, Kidney, and others.
- Segmentation (Auto Segmentation)
  - Spleen, Liver, Kidney, and others.

#### [Pathology](./pathology)
The pathology template includes example models for interactive and automated segmentation of pathology (WSI) images. It provides examples for the following two types of models:
- DeepEdit (Interactive + Auto Segmentation)
  - Nuclei Segmentation
- Segmentation (Auto Segmentation)
  - Nuclei multi-label segmentation for
    - Neoplastic cells
    - Inflammatory
    - Connective/Soft tissue cells
    - Dead Cells
    - Epithelial

#### [Endoscopy](./endoscopy)
The endoscopy template includes example models for interactive and automated tool tracking segmentation and a classification model for InBody vs. OutBody images in endoscopy-related images. It provides examples for the following three types of models:
- DeepEdit (Interactive + Auto Segmentation)
  - Tool Tracking
- Segmentation (Auto Segmentation)
  - Tool Tracking
- Classification
  - InBody vs. OutBody


#### [MONAI Bundle](./monaibundle)
The MONAI Bundle format provides a portable description of deep learning models. This template includes example models for interactive and automated segmentation using MONAI bundles defined in the MONAI Model Zoo. It can pull any bundle defined in the MONAI Model Zoo that is compatible and meets the requirements specified on the [MONAI Bundle Apps page](./monaibundle/).

### Creating a Custom App
Researchers may want to define and add their own models. Follow the steps below to add a new segmentation model:

1. Create a new TaskConfig file for the model in the `lib/configs` directory.
    - Use an existing TaskConfig file (such as `segmentation_spleen.py` in the radiology app, `tooltracking.py` in the endoscopy app, or `segmentation_nuclei.py` in the pathology app) as a reference.
     - In the new TaskConfig file, specify the model's network, labels, pretrained URL, and any other relevant attributes. Implement the following abstract classes:
        - `infer(self) -> Union[InferTask, Dict[str, InferTask]]`: Returns one or more Infer Tasks.
        - `trainer(self) -> Optional[TrainTask]`: Returns a TrainTask, or None if the model is an inference-only model. You can also accept any --conf <name> <value> and define the behavior of any function based on the new configuration.

2. Create a new Infer Task file for the model in the `lib/infers` directory.
    -  Use an existing Infer Task file (such as `segmentation_spleen.py`, `tooltracking.py`, or `segmentation_nuclei.py`) as a reference.
    - In the new Infer Task file, define pre- and post-transforms for the model.

3. Create a new Train Task file for the model in the `lib/trainers` directory.
    - Use an existing Train Task file (such as `segmentation_spleen.py`, `tooltracking.py`, or `segmentation_nuclei.py`) as a reference.
    - In the new Train Task file, define the loss function, optimizer, and pre- and post-transforms for the training and validation stages.

4. Run the app using the new model:
```bash
monailabel start_server --app workspace/<app_name> --studies workspace/images --conf models <model_name>
```

5. Replace `<app_name>` with the name of the app that you want to use (such as `radiology`, `endoscopy`, or `pathology`) and `<model_name>` with the name of the new segmentation model.

For development or debugging purposes, modify the `main()` function in `main.py` and run the train and infer tasks in headless mode:

```bash
export PYTHONPATH=workspace/<app_name>:$PYTHONPATH
python workspace/<app_name>/main.py
```

That's it! With these steps, you can add a new segmentation model to any of the MONAI Label apps.
