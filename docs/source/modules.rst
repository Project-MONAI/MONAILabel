================
Modules Overview
================

MONAI Label aims to allow researchers to build labeling applications in a serverless way.
This means that MONAI Label applications are always ready-to-deploy via MONAL Label server.

To develop a new MONAI labeling developers must inherit the `MONAILabelApp` and implement
the methods in the interface that are relevant to their labeling application. Typically a
labeling applications will consist of

- inferencing tasks used to allow end-users to invoke select pre-trained models,
- training tasks used to allow end-users to train an overlapping or completely different set of models,
- strategies that allow the users to select the next image to label.

Figure 1 shows the base set of interfaces that a developer may use to implement their app
and the various tasks their app should perform. For example, in the figure the user app `MyApp`
employs

- | two inferencing tasks, namely `MyInfer` which is a cusom implementation of `InferTask`, 
  | and `InferDeepGrow2D` which is a ready-to-use utilitiy included with MONAI Label,
- | one training task, `TrainDeepGrow` which is an extension of the `BasicTrainTask` utility,
- | and two next image selection strategies, `Random` included with MONAL Label which allow 
  | the user to select the next image at random, and `MyStrategy` which implements the interface 
  | `Strategy` which the end user may select as a custom alternative for next image selection

.. figure:: ../images/modules.svg
  :scale: 100%
  :alt: MONAI Label Interfaces and Utilities

  **Figure 1:** MONAI Label provides interfaces which can be implemented by the label app developer
  for custom functionality as well as utilities which are readily usable in the labeling app.

