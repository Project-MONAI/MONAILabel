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

    monailable --help

Downloading Sample Apps or Datasets
***********************************
You can download sample apps and datasets from *monailabel* CLI.

.. code-block::

  # Download Sample Apps
  monailabel apps # List sample apps
  monailabel apps --download --name deepedit_left_atrium --output apps

  # Download MSD Datasets
  monailabel datasets # List sample datasets
  monailabel datasets --download --name Task02_Heart --output datasets


Starting Server
***************
You can start server using *monailabel* CLI
::

  monailabel start_server --app apps\deepedit_left_atrium --studies datasets\Task02_Heart\imagesTr


.. note::

    Once you start the MONAI Label Server, by default it will be up and serving at http://127.0.0.1:8000/. Open the serving
    URL in browser. It will provide you the list of Rest APIs available.

3D Slicer Plugin
----------------
Download Preview Release from https://download.slicer.org/ and install MONAI Label plugin from Slicer Extension Manager.

Refer `3D Slicer plugin <https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer>`_ for other options to install and run MONAI Label plugin in 3D Slicer.
