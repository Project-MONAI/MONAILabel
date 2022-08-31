<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DeepLearning models for Endoscopy use-case(s).

> **_Development In Progress_**

![image](https://user-images.githubusercontent.com/7339051/182906351-4bb079dd-f211-450a-a526-ed753d05b093.png)

The App works best with [CVAT](https://github.com/opencv/cvat). Researchers/clinicians can place their studies in the local file folder.
1. Start with N images (for example total 100 unlabeled images)
2. MONAI Label computes the score and picks up the first batch of unlabeled images
   - For example server recommends 10 out of 100 (based on active learning - epistemic scoring
3. MONAI Label pushes and creates a project + task in CVAT for annotators to work upon
   - Project: **MONAILABEL**,  Task: **ActiveLearning_Iteration_1**
4. Annotator annotates and completes the task in CVAT
5. MONAI Label periodically checks if the Task is completed in CVAT
6. MONAI Label picks up those new samples and fine-tunes the existing model(s)
7. MONAI Label re-computes the score using the latest trained model and picks the next batch
   - For example server recommends another 10 of remaining 90 images
8. MONAI Label pushes and creates next task in CVAT for annotators to work upon
   - Project: **MONAILABEL**,  Task: **ActiveLearning_Iteration_2**
9. Cycle (Step 3 to Step 8) continues until you get some good enough model

#
Following is the summary of Active Learning workflow tests carried over roughly 4k samples.
> Using Active Learning strategy there is a big potential to reduce the number of annoations needed to train a good model.  The following table says, only 15% samples are good to label/annotate out of 4K unlabeled data to train a better model.

![image](https://user-images.githubusercontent.com/7339051/187589950-17c4476c-21d5-4ca5-a2f1-86681b667bfb.png)

#
Following snapshot shows Iteration cycles and progress for each Active Learning batch annotations in CVAT.

![image](../../docs/images/cvat_active_learning.jpeg)
> Publishing new fine-tuned model back to CVAT is not automated yet.  It will continue using pre-trained model.  Work is in progress...

### Structure of the App

- **[lib/infers](./lib/infers)** is the module where researchers define the inference class (i.e. type of inferer, pre
  transforms for inference, etc).
- **[lib/trainers](./lib/trainers)** is the module to define the pre and post transforms to train the network/model.
- **[lib/configs](./lib/configs)** is the module to define the image selection techniques.
- **[lib/transforms](./lib/transforms)** is the module to define customised transformations to be used in the App.
- **[lib/scoring](./lib/scoring)** is the module to define the image ranking techniques.
- **[main.py](./main.py)** is the script to extend [MONAILabelApp](../../monailabel/interfaces/app.py) class

Refer [How To Add New Model?](#how-to-add-new-model) section if you are looking to add your own model using this App as
reference.

### List of Pretrained Models

```bash
# List all the possible models
monailabel start_server --app /workspace/apps/radiology --studies /workspace/images
```

Following are the models which are currently added into Endosocpy App:

| Name                          | Description                                                                                                                                                                                                |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [deepedit](#deepedit)         | This model is based on DeepEdit: an algorithm that combines the capabilities of multiple models into one, allowing for both interactive and automated segmentation to label **Tool** among in-body images. |
| [tooltracking](#tooltracking) | A standard (non-interactive) segmentation model to label **Tool** among in-body images.                                                                                                                    |

> If both models are enabled, then Active Learning strategy uses [tooltracking](#tooltracking) model to rank the images.

### How To Use?

```bash
# skip this if you have already downloaded the app or using github repository (dev mode)
monailabel apps --download --name endoscopy --output workspace

# Pick DeepEdit model
monailabel start_server --app workspace/endoscopy --studies workspace/images --conf models deepedit

# Pick All
monailabel start_server --app workspace/endoscopy --studies workspace/images --conf models all

# Pick All
monailabel start_server --app workspace/endoscopy --studies workspace/images --conf models all

# Pick All + Preload into All GPU devices
monailabel start_server --app workspace/endoscopy --studies workspace/images --conf models all --conf preload true

# Pick All (Skip Training Tasks or Infer only mode)
monailabel start_server --app workspace/endoscopy --studies workspace/images --conf models all --conf skip_trainers true
```

#### With CVAT (Active Learning)
```bash
export MONAI_LABEL_DATASTORE=cvat
export MONAI_LABEL_DATASTORE_URL=http://127.0.0.1:8080
export MONAI_LABEL_DATASTORE_USERNAME=myuser
export MONAI_LABEL_DATASTORE_PASSWORD=mypass

monailabel start_server \
  --app workspace-apps/endoscopy \
  --studies workspace/images \
  --conf epistemic_enabled true \
  --conf epistemic_top_k 3 \
  --conf auto_finetune_models tooltracking \
  --conf auto_finetune_check_interval 30
```

Following are additional configs *(pass them as **--conf name value**) are useful when you use CVAT for Active Learning workflow.

| Name                         | Values | Default | Description                                                                                               |
|------------------------------|--------|---------|-----------------------------------------------------------------------------------------------------------|
| use_pretrained_model         | bool   | true    | Disable this NOT to load any pretrained weights                                                           |
| preload                      | bool   | false   | Preload model into GPU                                                                                    |
| skip_scoring                 | bool   | false   | Disable this to allow scoring methods to be used                                                          |
| skip_strategies              | bool   | false   | Disable this to add active learning strategies                                                            |
| epistemic_enabled            | bool   | false   | Enable Epistemic based Active Learning Strategy                                                           |
| epistemic_max_samples        | int    | 0       | Limit number of samples to run epistemic scoring (**_zero_** for no limit)                                |
| epistemic_simulation_size    | int    | 5       | Number of simulations per image to run epistemic scoring                                                  |
| epistemic_top_k              | int    | 10      | Select Top-K unlabeled for every active learning learning iteration                                       |
| auto_finetune_models         | str    |         | List of models to run fine-tuning when active learning task is completed  (**_None/Empty_** to train all) |
| auto_finetune_check_interval | int    | 60      | Interval in seconds for server to poll on **_CVAT_** to determine if active learning task is completed          |

### Model Overview

#### [DeepEdit](./lib/configs/deepedit.py)

This model based on DeepEdit. An algorithm that combines the capabilities of multiple models into one, allowing for both
interactive and automated segmentation.

This model is currently trained to segment **Tool** from 2D in-body images.

> monailabel start_server --app workspace/endoscopy --studies workspace/images --conf models deepedit

- Network: This App uses the [BasicUNet](https://docs.monai.io/en/latest/networks.html#basicunet) as the default network.
- Labels: `{ "Tool": 1 }`
- Dataset: The model is pre-trained over few in-body Images related to Endoscopy
- Inputs: 3 channels.
    - 1 channel for the image modality -> Automated mode
    - 2 channels (image modality + points for foreground and background clicks) -> Interactive mode
- Output: 1 channel representing the segmented Tool

#### [ToolTracking](./lib/configs/tooltracking.py)

This model based on UNet for automated segmentation. This model works for single label segmentation tasks.
> monailabel start_server --app workspace/endoscopy --studies workspace/images --conf models tooltracking

- Network: This model uses the [FlexibleUNet](https://docs.monai.io/en/latest/networks.html#flexibleunet) as the default network.
- Labels: `{ "Tool": 1 }`
- Dataset: The model is pre-trained over few in-body Images related to Endoscopy
- Inputs: 1 channel for the image modality
- Output: 1 channel representing the segmented Tool


### How To Add New Model?

Researches might want to define/add their own model(s). Or if there is a model as part of radiology use-case which is
generic and helpful for larger community, then you can follow the below steps to add a new model and using the same.

> As an example, you want to add new Segmentation model for **xyz**

- Create new TaskConfig **_segmentation_xyz.py_** in [lib/configs](./lib/configs).
    - Refer: [tooltracking.py](./lib/configs/tooltracking.py)
    - Fix attributes like network, labels, pretrained URL etc...
    - Implement abstract classes. Following are important ones.
        - `infer(self) -> Union[InferTask, Dict[str, InferTask]]` to return one or more Infer Task.
        - `trainer(self) -> Optional[TrainTask]` to return TrainTask. Return `None` if you are looking for Infer only
          model.
    - You can accept any `--conf <name> <value>` and define the behavior of any function based on new conf.
- Create new Infer Task **_segmentation_xyz.py_** in [lib/infers](./lib/infers).
    - Refer: [tooltracking.py](./lib/infers/tooltracking.py)
    - Importantly you will define pre/post transforms.
- Create new Train Task **_segmentation_xyz.py_** in [lib/trainers](./lib/trainers).
    - Refer: [tooltracking.py](./lib/trainers/tooltracking.py)
    - Importantly you will define loss_function, optimizer and pre/post transforms for training/validation stages.

- Run the app using new model
  > monailabel start_server --app workspace/endoscopy --studies workspace/images --conf models segmentation_xyz

For development or debugging purpose you can modify the **main()** function in [main.py](./main.py) and run train/infer
tasks in headless mode.

```bash
export PYTHONPATH=workspace/endoscopy:$PYTHONPATH
python workspace/endoscopy/main.py
```
