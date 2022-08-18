==================================
Bundle App Tutorial and Use Cases
==================================

Introduction: 
===============

Customized Networks for MONAI Label
------------------------------------

This tutorial introduces the usage of the Bundle app in MONAILabel  - **monaibundle**.  

The Bundle App empowers MONAILabel with customized models, pre- and post-processing, and any anatomies for labeling tasks. 
The Bundle App supports various training/inference frameworks with `Model-Zoo <https://github.com/Project-MONAI/model-zoo>`_

Highlights and Features:

  * Supporting customized models and networks such as SwinUNETR, AutoML, etc.
  * Advancing heterogeneous dataset (e.g., CT, MRI, Pathology, etc) with corresponding pre- and post-processing modules. 
  * Ready-to-Use inference of hundreds of anatomies (e.g., multi-organ abdominal segmentation, whole-brain segmentation) with trained model checkpoints.
  * Deploying robust interactive labeling tools such as DeepEdit.

.. _Model Zoo for MONAI Label:

Model Zoo for MONAI Label 
-----------------------------

MONAI Model Zoo hosts a collection of medical imaging models in the MONAI Bundle format. 
All source code of models (bundles) are tracked in models/, and for each distinct version of a bundle, 
it will be archived as a .zip file (named in the form of bundle_name_version.zip) and stored in Releases.

The monaibundle defines the model package and supports building python-based workflows via structured configurations

1. Self-contained model package with all the necessary information

2. Structured config that easy to override or reconstruct the workflow

3. Config provides good readability and usability by separating parameter settings from the python code

4. Config describes flexible workflow and components, allows for different low-level python implementations

Currently available bundles: `Model-Zoo <https://github.com/Project-MONAI/model-zoo>`_

.. _MONAI Label with 3DSlicer:

Prerequisite Setup
=================================

1. Install MONAI Label and 3DSlicer
-------------------------

For detailed setups of MONAILabel and 3D Slicer, refer to the `installation steps <https://docs.monai.io/projects/label/en/latest/installation.html>`_ guide 
if MONAILabel is not installed yet. 

2. Add MONAI Label Plugin in 3DSlicer
--------------------------------------

Add 3DSlicer with in-built MONAI Label plugin if not setup yet. Refer to **Step 3** 
in `installation <https://docs.monai.io/projects/label/en/latest/installation.html>`_ guide.

.. _Select Bundle and Load Configuration to MONAI Label:

Use Case 1: Bundle for SwinUNETR Multi-Organ Segmentation
================================================================================

On the local machine follow the commands listed below to install MONAI Label, and deploy the bundle app and standard dataset on the MONAI Label server.

* Step 1: Install and start MONAI Label server with the Bundle app

.. code-block:: bash

  # install MONAI Label
  pip install monailabel

  # download Bundle sample app to local directory
  monailabel apps --name monaibundle --download --output .

  # download a local study images, sample dataset such as spleen:
  monailabel datasets --download --name Task09_Spleen --output .

  # start the bundle app in MONAI label server
  # and start annotating the images using bundle with the Swin UNETR bundle
  monailabel start_server --app monaibundle --studies Task09_Spleen/imagesTr --conf models swin_unetr_btcv_segmentation_v0.1.0


* Step 2: Start 3DSlicer

* Step 3: Start the SwinUNETR bundle and follow clicks

- On the menu bar navigate click **MONAI Label** 

  .. image:: ../images/quickstart/1.jpeg
    :alt: 3DSlicer setup
    :width: 800

- Check the Model Zoo loading, MONAI Bundle app, and load study image.

  .. image:: ../images/quickstart/2.jpeg
    :alt: load data
    :width: 800

- Select bundle models and obtain automatic labels

  .. image:: ../images/quickstart/3.jpeg
    :alt: inference
    :width: 800

Now get the automatic inference of the trained SwinUNETR model!

- Submit refined labels and train to fine-tune the model. 

  .. image:: ../images/quickstart/4.jpeg
    :alt: inference
    :width: 800

.. |MLIcon| image:: ../images/quickstart/MONAILabel.png
  :width: 20

Use Case 2: Bundle with Customized Scripts for Renal Substructure Segmentation
=================================================================================

This use case provides an instruction on using bundle model with customized scripts. 


.. code-block:: bash

  # install MONAI Label
  pip install monailabel

  # download Bundle sample app to local directory
  monailabel apps --name monaibundle --download --output .

  # download a local study images, sample dataset such as spleen:
  monailabel datasets --download --name Task09_Spleen --output .
 
  # download the bundle and save to the monaibundle/model and direct to the customized bundle folder
  cd <path to the bundle model>/renalStructures_UNEST_segmentation_v0.1.0

  # add customized scripts in the downloaded bundle
  export PYTHONPATH=$PYTHONPATH:"'/monaibundle/model/renalStructures_UNEST_segmentation_v0.1.0/scripts"

  # start the bundle app in MONAI label server 
  monailabel start_server --app <full path to the monaibundle app/monaibundle> --studies <path to the local dataset/Task09_Spleen/imagesTr> 
  --conf models renalStructures_UNEST_segmentation_v0.1.0



- Start 3D Slicer and follow same MONAI Label plugin process **MONAI Label** 

- Select the customized bundle and inference with pre-trained model for renal structure segmentation

  .. image:: ../images/quickstart/5.jpeg
    :alt: 3DSlicer setup
    :width: 800

Get inferred label with renal cortex, medulla, and collecting system.