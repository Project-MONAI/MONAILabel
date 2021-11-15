========================
Installation
========================

Prerequisites
---------------
MONAI Label supports following OS with GPU/CUDA enabled.

Windows
********
Make sure you have python 3.x version environment with PyTorch and CUDA installed.

- Install `Python <https://www.python.org/downloads/>`_
- Install the following Python libraries

.. code-block::

    python -m pip install --upgrade pip setuptools wheel

    # Install latest stable version for pytorch
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

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
  monailabel apps --download --name deepedit --output apps

  # Download MSD Datasets
  monailabel datasets # List sample datasets
  monailabel datasets --download --name Task09_Spleen --output datasets


Starting Server
***************
You can start server using *monailabel* CLI
::

  monailabel start_server --app apps\deepedit --studies datasets\Task09_Spleen\imagesTr


.. note::

    Once you start the MONAI Label Server, by default it will be up and serving at http://127.0.0.1:8000/. Open the serving
    URL in browser. It will provide you the list of Rest APIs available.

Deployment
----------
MONAI Label Server uses `Uvicorn <https://www.uvicorn.org/>`_ which is a lightning-fast ASGI server implementation.
However user can deploy the application in any server that supports `ASGI specification <https://asgi.readthedocs.io/en/latest/>`_

There are `multiple choices <https://www.uvicorn.org/deployment/>`_ available for Uvicorn to run as Development Server vs Standalone Server vs Production.

Deploying MONAI Label server for production use is out of project scope.  However for basic production deployment, you might need to run Uvicorn independently.  In such cases, you can following these simple steps.

::

  # dryrun the MONAI Label CLI for pre-init and dump the env variables to .env or env.bat
  monailabel start_server --app apps\deepedit --studies datasets\Task09_Spleen\imagesTr --host 0.0.0.0 --port 8000 --dryrun

  # Linux/Ubuntu
  source .env
  uvicorn monailabel.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-config apps\deepedit\logs\logging.json \
    --no-access-log


  # Windows
  call env.bat
  uvicorn monailabel.app:app ^
    --host 0.0.0.0 ^
    --port 8000 ^
    --log-config apps\deepedit\logs\logging.json ^
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

  monailabel start_server --app apps\deepedit --studies http://127.0.0.1:8042/dicom-web


.. note::

    Please install `Orthanc <https://www.orthanc-server.com/download.php>`_ before using OHIF Viewer.

    For Ubuntu 20.x, Orthanc can be installed as `apt-get install orthanc orthanc-dicomweb`.
    However, you have to **upgrade to latest version** by following steps mentioned `here <https://book.orthanc-server.com/users/debian-packages.html#replacing-the-package-from-the-service-by-the-lsb-binaries>`_

    You can use `PlastiMatch <https://plastimatch.org/plastimatch.html#plastimatch-convert>`_ to convert NIFTI to DICOM

    OHIF Viewer will be accessible at http://127.0.0.1:8000/ohif/
