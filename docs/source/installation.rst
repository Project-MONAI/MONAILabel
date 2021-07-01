========================
MONAI Label Installation
========================

-------------------
Ubuntu Installation
-------------------

.. code:: bash

  # One time setup (to pull monai with nvidia gpus)
  docker run -it --rm --gpus all --ipc=host --net=host -v /rapid/xyz:/workspace/ projectmonai/monai:0.5.2
  
  # Install monailabel 
  pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel
  
  # Download MSD Datasets
  monailabel datasets # list sample datasets
  monailabel datasets --download --name Task02_Heart --output /workspace/datasets/
  
  # Download Sample Apps
  monailabel apps # list sample apps
  monailabel apps --download --name deepedit_left_atrium --output /workspace/apps/
  
  # Start Server
  monailabel start_server --app /workspace/apps/deepedit_left_atrium --studies /workspace/datasets/Task02_Heart/imagesTr

--------------------
Windows Installation
--------------------

Prerequisites
-------------

Make sure you have python 3.x version environment with PyTorch and CUDA installed.

- Install `Python <https://www.python.org/downloads/>`_
- Install `CUDA <https://developer.nvidia.com/cuda-downloads>`_ (Faster mode: install CUDA runtime only)
- Install the following Python libraries

  .. code:: Python
  
    python -m pip install --upgrade pip setuptools wheel
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    python -c "import torch; print(torch.cuda.is_available())"

Installing MONAI Label
----------------------

.. code:: python

  pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel
  monailabel -h
  
  # Download MSD Datasets
  monailabel datasets # List sample datasets
  monailabel datasets --download --name Task02_Heart --output C:\Workspace\Datasets
  
  # Download Sample Apps
  monailabel apps # List sample apps
  monailabel apps --download --name deepedit_left_atrium --output C:\Workspace\Apps
  
  # Start Server
  monailabel start_server --app C:\Workspace\Apps\deepedit_left_atrium --studies C:\Workspace\Datasets\Task02_Heart\imagesTr


> Once you start the MONAILabel Server, by default it will be up and serving at http://127.0.0.1:8000/. Open the serving
  URL in browser. It will provide you the list of Rest APIs available.

----------------------------
3DSlicer Module Installation
----------------------------

Refer `3D Slicer plugin <plugins/slicer>`_ for installing and running MONAILabel plugin in 3D Slicer.
