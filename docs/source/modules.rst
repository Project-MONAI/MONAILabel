================
Modules Overview
================

MONAI Label aims to allow researchers to build labeling applications in a serverless way.
This means that MONAI Label applications are always ready-to-deploy via MONAL Label server.

To develop a new MONAI labeling app, developers must inherit the :py:class:`~monailabel.interfaces.MONAILabelApp` interface
and implement the methods in the interface that are relevant to their application. Typically a
labeling applications will consist of

- inferencing tasks to allow end-users to invoke select pre-trained models,
- training tasks used to allow end-users to train a set of models,
- strategies that allow the users to select the next image to label based on some criteria.

Figure 1 shows the base interfaces that a developer may use to implement their app
and the various tasks their app may perform. For example, in the figure the user app :py:class:`MyApp`
employs

- | two inferencing tasks, namely :py:class:`MyInfer` which is a custom implementation of :py:class:`~monailabel.interfaces.tasks.InferTask`, 
  | and :py:class:`~monailabel.utils.infer.deepgrow_2d.InferDeepGrow2D` which is a ready-to-use utility included with MONAI Label,
- one training task, :py:class:`TrainDeepGrow` which is an extension of the :py:class:`~monailabel.utils.train.base_train.BasicTrainTask` utility,
- | and two next image selection strategies, :py:class:`~monailabel.interfaces.utils.activelearning.Random` included with MONAL Label which allow 
  | the user to select the next image at random, and :py:class:`MyStrategy` which implements the interface 
  | :py:class:`~monailabel.interfaces.Strategy` which the end user may select as a custom alternative for next image selection

.. figure:: ../images/modules.svg
  :alt: MONAI Label Interfaces and Utilities

  **Figure 1:** MONAI Label provides interfaces which can be implemented by the label app developer
  for custom functionality as well as utilities which are readily usable in the labeling app.


In next few sections we will go into the details of implementing 

- `an inference task <#building-an-inference-task>`_,
- `a training task <#building-a-training-task>`_,
- `an image selection strategy <#build-an-image-selection-strategy>`_,

and putting these to work together in a `MONAI Label app <#building-a-monai-label-app>`_.

Building an Inference Task
================================

Inference tasks must implement the :py:class:`~monailabel.interfaces.InferTask` interface where one must specify a list of pre- and post-transforms
and an inferer model. The code snippet below is an example implementation of :py:class:`~monailabel.interfaces.InferTask` where the image is pre-processed
to a Numpy array, input into :py:class:`SimpleInferer`, and the result is post-processed by applying sigmoid activation with binary
discretization.

.. code-block:: python
  :emphasize-lines: 7, 9, 15, 18

  from monai.inferers import SimpleInferer
  from monai.transforms import (LoadImaged, ToNumpyd, Activationsd
                                AsDiscreted, ToNumpyd)

  from monailabel.interfaces.tasks import InferTask

  class MyInfer(InferTask):

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            ToNumpyd(keys="image"),
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
            ToNumpyd(keys="pred"),
        ]

Building a Training Task
===============================

Training tasks may extend the base class :py:class:`~monailabel.utils.train.basic_train.BasicTrainTask` which is an abstraction over supervised trainers and evaluators.
Here, the developer may override the functionality of the base training class with the desired behavior.

The code block below shows a sample implementation specifying the loss function, training pre- and post-transforms, and validation 
pre-transforms and inference. There are many more aspects of :py:class:`~monailabel.utils.train.basic_train.BasicTrainTask` that the developer may choose to override, but
in this example they follow the default behavior in the base class.

.. code-block:: python
  :emphasize-lines: 6, 8, 11, 19, 25, 34

  from monai.inferers import SlidingWindowInferer
  from monai.transforms import *

  from monailabel.utils.train.basic_train import BasicTrainTask

  class MyTrainTask(BasicTrainTask):

    def loss_function(self):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def train_pre_transforms(self):
        return Compose([
            LoadImaged(keys=("image", "label")),
            AsChannelFirstd(keys=("image", "label")),
            SpatialCropForegroundd(keys=("image", "label"), source_key="label", spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image"),
        ])

    def train_post_transforms(self):
        return Compose([
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
        ])

    def val_pre_transforms(self):
        return Compose([
            LoadImaged(keys=("image", "label")),
            AsChannelFirstd(keys=("image", "label")),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            ToTensord(keys=("image", "label")),
        ])

    def val_inferer(self):
        return SlidingWindowInferer(roi_size=(128, 128, 128))


Building an Image Selection Strategy
====================================

Selecting the next image to load in the end-users client may be of importance to some labeling
applications where the developer may want to allow the user to select one (of perhaps many)
strategies to select the next image to annotate as a means to efficiently annotate the datastore
by, for example, presenting the most representative image of an unlabeled subset of images.

The example below shows a simple image selection strategy where :py:class:`GetFirstUnlabeledImage` returns
the first unlabeled image it finds in the :py:class:`~monailabel.interfaces.Datastore`.

.. code-block:: python
  :emphasize-lines: 6, 8

  import logging

  from monailabel.interfaces import Datastore
  from monailabel.interfaces.tasks import Strategy

  class GetFirstUnlabeledImage(Strategy):

      def __call__(self, request, datastore: Datastore):
          images = datastore.get_unlabeled_images()
          if not len(images):
              return None

          images.sort()
          image = images[0]

          return image


Building a MONAI Label App
==========================

A MONAI Label app ties together inference, training, and image selection to provide the end-user with
a seamless simultaneous model training and annotation experience, where a segmentation model learns
how to segment the region of interest as the user annotates the data.

The labeling app in the example code below utilizes the tasks :py:class:`MyInfer`, :py:class:`MyTrain`,
and :py:class:`MyStrategy` we have defined so far. In this example, we define a :py:class:`UNet` architecture
where we load pre-trained the model weights, an use them or inference. During training, we load the model
weights perform training based on newly annotated data, save the new weights and use them in future inferences.
Any further training on the model will continue from the latest checkpoint.

.. code-block:: python
  :emphasize-lines: 7, 9, 21, 26, 66

  from monai.networks.layers import Norm
  from monai.networks.nets import UNet

  from monailabel.interfaces import MONAILabelApp
  from monailabel.utils.activelearning import Random

  import MyInfer, MyTrain, GetFirstUnlabeledImage

  class MyApp(MONAILabelApp):

      def __init__(self, app_dir, studies):
          self.model_dir = os.path.join(app_dir, "model")
          self.network = UNet(...)

          self.pretrained_model = os.path.join(self.model_dir, "segmentation_spleen.pt")
          self.final_model = os.path.join(self.model_dir, "final.pt")
          self.train_stats_path = os.path.join(self.model_dir, "train_stats.json")

          path = [self.pretrained_model, self.final_model]
          infers = {
              "segmentation_spleen": MyInfer(path, self.network),
          }

          strategies = {
              "random": Random(),
              "first": GetFirstUnlabeledImage(),
          }

          resources = [
              (
                  self.pretrained_model,
                  "https://api.ngc.nvidia.com/v2/models/nvidia/med"
                  "/clara_pt_spleen_ct_segmentation/versions/1/files/models/model.pt",
              ),
          ]

          super().__init__(
              app_dir=app_dir,
              studies=studies,
              infers=infers,
              strategies=strategies,
              resources=resources,
          )

          # Simple way to Add deepgrow 2D+3D models for infer tasks
          self.add_deepgrow_infer_tasks()

      def train(self, request):
          output_dir = os.path.join(self.model_dir, request.get("name", "model_01"))

          # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
          load_path = os.path.join(output_dir, "model.pt")

          # Use pretrained weights to start training
          load_path = (
              load_path
              if os.path.exists(load_path)
              else self.pretrained_model
              if request.get("pretrained", True)
              else None
          )

          # Datalist for train/validation
          train_d, val_d = self.partition_datalist(self.datastore().datalist(), request.get("val_split", 0.2))

          task = MyTrain(
              output_dir=output_dir,
              train_datalist=train_d,
              val_datalist=val_d,
              network=self.network,
              load_path=load_path,
              publish_path=self.final_model,
              stats_path=self.train_stats_path,
              device=request.get("device", "cuda"),
              lr=request.get("lr", 0.0001),
              val_split=request.get("val_split", 0.2),
              max_epochs=request.get("epochs", 1),
              amp=request.get("amp", True),
              train_batch_size=request.get("train_batch_size", 1),
              val_batch_size=request.get("val_batch_size", 1),
          )
          return task()

      def train_stats(self):
          if os.path.exists(self.train_stats_path):
              with open(self.train_stats_path, "r") as fc:
                  return json.load(fc)
          return super().train_stats()

