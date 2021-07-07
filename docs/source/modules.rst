================
Modules Overview
================

MONAI Label aims to allow researchers to build labeling applications in a serverless way.
This means that MONAI Label applications are always ready-to-deploy via MONAL Label server.

To develop a new MONAI labeling developers must inherit the ``MONAILabelApp`` and implement
the methods in the interface that are relevant to their labeling application. Typically a
labeling applications will consist of

- inferencing tasks used to allow end-users to invoke select pre-trained models,
- training tasks used to allow end-users to train an overlapping or completely different set of models,
- strategies that allow the users to select the next image to label.

Figure 1 shows the base set of interfaces that a developer may use to implement their app
and the various tasks their app should perform. For example, in the figure the user app `MyApp`
employs

- | two inferencing tasks, namely ``MyInfer`` which is a cusom implementation of ``InferTask``, 
  | and ``InferDeepGrow2D`` which is a ready-to-use utilitiy included with MONAI Label,
- one training task, ``TrainDeepGrow`` which is an extension of the ``BasicTrainTask`` utility,
- | and two next image selection strategies, ``Random`` included with MONAL Label which allow 
  | the user to select the next image at random, and ``MyStrategy`` which implements the interface 
  | ``Strategy`` which the end user may select as a custom alternative for next image selection

.. figure:: ../images/modules.svg
  :alt: MONAI Label Interfaces and Utilities

  **Figure 1:** MONAI Label provides interfaces which can be implemented by the label app developer
  for custom functionality as well as utilities which are readily usable in the labeling app.


In next few sections we will go into the details of implementing inferece an trainging tasks, and 
putting these tasks to work together in a MONAI Label app. We will go into
some detail on how to build

- `a custom inference task <#building-a-custom-inference-task>`_,
- `a custom training task <#building-a-custom-training-task>`_,
- `a labeling app <#building-a-custom-monai-label-app>`_.

Building a Custom Inference Task
================================

Custom inference tasks must implement the ``InferTask`` interface where

- ``pre_transforms`` and ``post_transforms`` return a ``List`` of callables (e.g. MONAI transforms),
- ``inferer`` returns a callable (e.g. a MONAI inferer).

Below, is an example implementation of ``InferTask`` where

- | ``pre_transforms`` returns a chain of transformations which load images and converting them into a Numpy arrays
  | (as preparation for inference),
- ``inferer`` returns an instance of MONAI's ``SimpleInferer``,
- ``post_transforms`` returns a chain of transformations which convert the values of the inference into activations.

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

Building a Custom Training Task
===============================

Custom training tasks must implement the ``TrainTask`` interface, or extend the partial functionality
of the ``BasicTrainTask`` class. implementation of ``TrainTask`` requires the developer to specify all
implemnetational aspects of training which can be work-intensive, but may be necessary in special or
unique circumstances. However, for a large part of use cases the developer only need extend
``BasicTrainTask`` provides the core functionality for training and validation and the develop only need
specify a small subset of the training implementation details.

The code block below shows a sample custom implementation, ``MyTrainTask``, of the training task from ``BasicTrainTask``
where the custom training task specifies

- ``loss_function`` which returns the training loss,
- ``train_pre_transforms`` and ``train_post_transforms`` which return a ``Compose`` of callables (e.g. MONAI transforms),
- ``val_pre_transforms`` which return a ``Compose`` of callables which prepare the data for validation inference,
- ``val_inferer`` which returns a callable which runs inference on the data processed by ``val_pre_transforms``.

.. code-block:: python
  :emphasize-lines: 6, 8, 11, 20, 28, 39

  from monai.inferers import SlidingWindowInferer
  from monai.transforms import *

  from monailabel.utils.train.basic_train import BasicTrainTask

  class MyTrainTask(BasicTrainTask):

    def loss_function(self):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def train_pre_transforms(self):
        t = [
            LoadImaged(keys=("image", "label")),
            AsChannelFirstd(keys=("image", "label")),
            SpatialCropForegroundd(keys=("image", "label"), source_key="label", spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image"),
        ]
        return Compose(t)

    def train_post_transforms(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
            ]
        )

    def val_pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                AsChannelFirstd(keys=("image", "label")),
                ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=("image", "label"), source_key="image"),
                ToTensord(keys=("image", "label")),
            ]
        )

    def val_inferer(self):
        return SlidingWindowInferer(roi_size=(128, 128, 128))


Building a Custom MONAI Label App
=================================


