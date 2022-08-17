.. comment
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


:github_url: https://github.com/Project-MONAI/MONAILabel

=============
API Reference
=============

MONAILabel APP
==============


.. currentmodule:: monailabel.interfaces.app
.. autoclass:: MONAILabelApp
    :members:
    :noindex:

.. currentmodule:: monailabel.interfaces.datastore
.. autoclass:: Datastore
    :members:
    :noindex:

.. currentmodule:: monailabel.interfaces.exception
.. autoclass:: MONAILabelError
    :members:
    :noindex:
.. autoclass:: MONAILabelException
    :members:
    :noindex:

Tasks
=====

.. currentmodule:: monailabel.interfaces.tasks.infer
.. autoclass:: InferType
    :members:
    :noindex:
.. autoclass:: InferTask
    :members:
    :noindex:

.. currentmodule:: monailabel.interfaces.tasks.train
.. autoclass:: TrainTask
    :members:
    :noindex:

.. currentmodule:: monailabel.interfaces.tasks.strategy
.. autoclass:: Strategy
    :members:
    :noindex:

.. currentmodule:: monailabel.interfaces.tasks.scoring
.. autoclass:: ScoringMethod
    :members:
    :noindex:

Utils
=====

.. currentmodule:: monailabel.tasks.train.basic_train
.. autoclass:: BasicTrainTask
    :members:
    :noindex:


Client
======

.. currentmodule:: monailabel.client.client
.. autoclass:: MONAILabelClient
    :members:
    :noindex:
.. autoclass:: MONAILabelError
    :members:
    :noindex:

Modules
=======

.. toctree::
    :maxdepth: 4

    apidocs/modules
