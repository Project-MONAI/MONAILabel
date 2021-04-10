:github_url: https://github.com/Project-MONAI/MONAI-Label

.. _apis:

MONAI APP
=========


.. automodule:: monailabel.interface.app
.. autoclass:: MONAILabelApp
    :members:

Others
======

.. automodule:: monailabel.interface.exception
.. autoclass:: MONAILabelError
    :members:
.. autoclass:: MONAILabelException
    :members:

.. automodule:: monailabel.interface.infer
.. autoclass:: InferenceEngine
    :members:

.. automodule:: monailabel.interface.train
.. autoclass:: TrainEngine
    :members:


Engines
=======

Inference
---------
.. automodule:: monailabel.engines.infer.deepgrow_2d
.. autoclass:: Deepgrow2D

.. automodule:: monailabel.engines.infer.deepgrow_3d
.. autoclass:: Deepgrow3D

.. automodule:: monailabel.engines.infer.segmentation_spleen
.. autoclass:: SegmentationSpleen

Train
-----
.. automodule:: monailabel.engines.train.segmentation_spleen
.. autoclass:: SegmentationSpleen

