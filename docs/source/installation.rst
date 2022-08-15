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


========================
Installation
========================

Prerequisites
---------------
MONAI Label supports both **Ubuntu** and **Windows** OS with GPU/CUDA enabled.

Make sure you have python 3.7/3.8/3.9 version environment with PyTorch and CUDA installed.  MONAI Label features on other python version are not verified.

- Install `Python <https://www.python.org/downloads/>`_
- Install the following Python libraries

.. code-block::

    python -m pip install --upgrade pip setuptools wheel

    # Install latest stable version for pytorch
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

    # Check if cuda enabled
    python -c "import torch; print(torch.cuda.is_available())"


Install From PyPI
-----------------

Milestone release
*****************

To install the `current milestone release <https://pypi.org/project/monailabel/>`_:
::

    pip install monailabel

Weekly preview release
**********************
To install the `weekly preview release <https://pypi.org/project/monailabel-weekly/>`_:
::

    pip install monailabel-weekly

The weekly build is released to PyPI every Sunday with a pre-release build number *dev[%y%U]*.

From GitHub
***********
To install latest from github main branch
::

    pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel

.. note::

    If you have installed the
    PyPI release version using ``pip install monailabel``, please run ``pip uninstall
    monailabel`` before using the commands from this section. Because ``pip`` by
    default prefers the milestone release.

The milestone versions are currently planned and released every few months.  As the
codebase is under active development, you may want to install MONAI from GitHub
for the latest features

From DockerHub
**************
To install latest from `DockerHub <https://hub.docker.com/r/projectmonai/monailabel>`_:
::

    docker run -it --rm --gpus all --ipc=host --net=host -v ~:/workspace/ projectmonai/monailabel:latest bash


MONAI Label CLI
---------------
Simple *monailabel* command will help user to download sample apps, datasets and run server.
::

    monailabel --help

Downloading Sample Apps or Datasets
***********************************
You can download sample apps and datasets from *monailabel* CLI.

.. code-block::

  # Download Sample Apps
  monailabel apps # List sample apps
  monailabel apps --download --name radiology --output apps

  # Download MSD Datasets
  monailabel datasets # List sample datasets
  monailabel datasets --download --name Task09_Spleen --output datasets


Starting Server
***************
You can start server using *monailabel* CLI
::

  # Run Deepedit Model.
  # Options can be (deepedit|deepgrow|segmentation|segmentation_spleen|all) in case of radiology app.
  # You can also pass comma seperated models like --conf models deepedit,segmentation

  monailabel start_server --app apps/radiology --studies datasets/Task09_Spleen/imagesTr --conf models deepedit


.. note::

    Once you start the MONAI Label Server, by default it will be up and serving at http://127.0.0.1:8000/. Open the serving
    URL in browser. It will provide you the list of Rest APIs available.

Deployment
----------
MONAI Label Server uses `Uvicorn <https://www.uvicorn.org/>`_ which is a lightning-fast ASGI server implementation.
However user can deploy the application in any server that supports `ASGI specification <https://asgi.readthedocs.io/en/latest/>`_

There are `multiple choices <https://www.uvicorn.org/deployment/>`_ available for Uvicorn to run as Development Server vs Standalone Server vs Production.

Deploying MONAI Label server for production use is out of project scope.

Run MONAI Label server in ssl mode:
***********************************
You can run MONAILabel server in https mode.
.. code-block::

  # Create self-signed ssl cert
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout uvicorn-selfsigned.key -out uvicorn-selfsigned.crt

  # Start server in ssl mode
  monailabel start_server --app apps/radiology --studies datasets/Task09_Spleen/imagesTr --conf models deepedit --ssl_keyfile uvicorn-selfsigned.key --ssl_certfile uvicorn-selfsigned.crt



However for basic production deployment, you might need to run Uvicorn independently.  In such cases, you can following these simple steps.

::

  # dryrun the MONAI Label CLI for pre-init and dump the env variables to .env or env.bat
  monailabel start_server --app apps/radiology --studies datasets/Task09_Spleen/imagesTr --host 0.0.0.0 --port 8000 --dryrun

  # Linux/Ubuntu
  source .env
  uvicorn monailabel.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-config apps/radiology/logs/logging.json \
    --no-access-log


  # Windows
  call env.bat
  uvicorn monailabel.app:app ^
    --host 0.0.0.0 ^
    --port 8000 ^
    --log-config apps\radiology\logs\logging.json ^
    --no-access-log


For more options about Uvicorn (concurrency, SSL etc..) refer: https://www.uvicorn.org/#command-line-options

3D Slicer Plugin
----------------
Download Preview Release from https://download.slicer.org/ and install MONAI Label plugin from Slicer Extension Manager.

Refer `3D Slicer plugin <https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer>`_ for other options to install and run MONAI Label plugin in 3D Slicer.

.. note::

    To avoid accidentally using an older Slicer version, you may want to *uninstall* any previously installed 3D Slicer package.

OHIF Plugin
-----------
MONAI Label comes with `pre-built plugin <https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/ohif>`_ for `OHIF Viewer <https://github.com/OHIF/Viewers>`_.  To use OHIF Viewer, you need to provide DICOMWeb instead of FileSystem as *studies* when you start the server.

::

  monailabel start_server --app apps/radiology --studies http://127.0.0.1:8042/dicom-web --conf models deepedit


If you have authentication set for dicom-web then you can pass the credentials using environment `variables <https://github.com/Project-MONAI/MONAILabel/blob/main/monailabel/config.py>`_ while running the server.

::

  export MONAI_LABEL_DICOMWEB_USERNAME=xyz
  export MONAI_LABEL_DICOMWEB_PASSWORD=abc
  monailabel start_server --app apps/radiology --studies http://127.0.0.1:8042/dicom-web --conf models deepedit


.. note::

    Please install `Orthanc <https://www.orthanc-server.com/download.php>`_ before using OHIF Viewer.

    For Ubuntu 20.x, Orthanc can be installed as `apt-get install orthanc orthanc-dicomweb`.
    However, you have to **upgrade to latest version** by following steps mentioned `here <https://book.orthanc-server.com/users/debian-packages.html#replacing-the-package-from-the-service-by-the-lsb-binaries>`_

    You can use `PlastiMatch <https://plastimatch.org/plastimatch.html#plastimatch-convert>`_ to convert NIFTI to DICOM

    OHIF Viewer will be accessible at http://127.0.0.1:8000/ohif/

QuPath
-------
For pathology usecase, you can install `QuPath <https://qupath.github.io/>`_ and basic monailabel extension in QuPath.
You can download sample whole slide images
from `https://portal.gdc.cancer.gov/repository <https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Slide%20Image%22%5D%7D%7D%5D%7D>`_

::

  # start server using pathology over downloaded whole slide images
  monailabel start_server --app apps/pathology --studies wsi_images


Refer `QuPath Plugin <https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/qupath>`_ for installing and running MONAILabel plugin in QuPath.


Digital Slide Archive (DSA)
---------------------------
If you have `DSA <https://digitalslidearchive.github.io/digital_slide_archive/>`_ setup running,  you can use the same for annotating Pathology images using MONAILabel.

::

  # start server using pathology connecting to DSA server
  monailabel start_server --app apps/pathology --studies http://0.0.0.0:8080/api/v1

Refer `DSA Plugin <https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/dsa>`_ for running a sample pathology use-case in MONAILabel using DSA.
