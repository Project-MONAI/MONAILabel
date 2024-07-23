#################################
#
# Imports from useful Python libraries
#
#################################

import numpy
import os
import monai
import skimage
import importlib.metadata
import subprocess
import uuid
import shutil
import tempfile
import logging
import sys

#################################
#
# Imports from CellProfiler
#
##################################

from cellprofiler_core.image import Image
from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Binary, ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.preferences import get_default_output_directory
from cellprofiler_core.setting.text import (
    Integer,
    ImageName,
    Directory,
    Filename,
    Float,
)

CUDA_LINK = "https://pytorch.org/get-started/locally/"
Cellpose_link = " https://doi.org/10.1038/s41592-020-01018-x"
Omnipose_link = "https://doi.org/10.1101/2021.11.03.467199"
LOGGER = logging.getLogger(__name__)

__doc__ = f"""\
RunCellpose
===========

**RunCellpose** uses a pre-trained machine learning model (Cellpose) to detect cells or nuclei in an image.

This module is useful for automating simple segmentation tasks in CellProfiler.
The module accepts greyscale input images and produces an object set. Probabilities can also be captured as an image.

Loading in a model will take slightly longer the first time you run it each session. When evaluating
performance you may want to consider the time taken to predict subsequent images.

This module now also supports Ominpose. Omnipose builds on Cellpose, for the purpose of **RunCellpose** it adds 2 additional
features: additional models; bact-omni and cyto2-omni which were trained using the Omnipose architechture, and bact
and the mask reconstruction algorithm for Omnipose that was created to solve over-segemnation of large cells; useful for bacterial cells,
but can be used for other arbitrary and anisotropic shapes. You can mix and match Omnipose models with Cellpose style masking or vice versa.

The module has been updated to be compatible with the latest release of Cellpose. From the old version of the module the 'cells' model corresponds to 'cyto2' model.

Installation:

It is necessary that you have installed Cellpose version >= 1.0.2

You'll want to run `pip install cellpose` on your CellProfiler Python environment to setup Cellpose. If you have an older version of Cellpose
run 'python -m pip install cellpose --upgrade'.

To use Omnipose models, and mask reconstruction method you'll want to install Omnipose 'pip install omnipose' and Cellpose version 1.0.2 'pip install cellpose==1.0.2'.

On the first time loading into CellProfiler, Cellpose will need to download some model files from the internet. This
may take some time. If you want to use a GPU to run the model, you'll need a compatible version of PyTorch and a
supported GPU. Instructions are avaiable at this link: {CUDA_LINK}

Stringer, C., Wang, T., Michaelos, M. et al. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021). {Cellpose_link}
Kevin J. Cutler, Carsen Stringer, Paul A. Wiggins, Joseph D. Mougous. Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. bioRxiv 2021.11.03.467199. {Omnipose_link}
============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

"Select Cellpose Docker Image"
CELLPOSE_DOCKER_NO_PRETRAINED = "cellprofiler/runcellpose_no_pretrained:0.1"
CELLPOSE_DOCKER_IMAGE_WITH_PRETRAINED = "cellprofiler/runcellpose_with_pretrained:0.1"

"Detection mode"
MODEL_NAMES = ['cyto','nuclei','tissuenet','livecell', 'cyto2', 'general',
                'CP', 'CPx', 'TN1', 'TN2', 'TN3', 'LC1', 'LC2', 'LC3', 'LC4', 'custom']

"Bundle save path"
BUNDLE_PATH = os.path.join(tempfile.gettempdir(), "bundles")


class RunVISTA2D(ImageSegmentation):
    category = "Object Processing"

    module_name = "RunVISTA2D"

    variable_revision_number = 1

    doi = {
        "Please cite the following when using RunVISTA2D:": "https://doi.org/10.48550/arXiv.2406.05285",
    }

    VISTA2D_BUNDLE_NAME = "cell_vista_segmentation"
    VISTA2D_BUNDLE_VERSION = "0.2.1"

    def create_settings(self):
        super(RunVISTA2D, self).create_settings()

        self.docker_or_python = Choice(
            text="Run CellPose in docker or local python environment",
            choices=["Docker", "Python"],
            value="Docker",
            doc="""\
If Docker is selected, ensure that Docker Desktop is open and running on your
computer. On first run of the RunCellpose plugin, the Docker container will be
downloaded. However, this slow downloading process will only have to happen
once.

If Python is selected, the Python environment in which CellProfiler and Cellpose
are installed will be used.
""",
        )

        self.use_gpu = Binary(
            text="Use GPU",
            value=False,
            doc=f"""\
If enabled, Cellpose will attempt to run detection on your system's graphics card (GPU).
Note that you will need a CUDA-compatible GPU and correctly configured PyTorch version, see this link for details:
{CUDA_LINK}

If disabled or incorrectly configured, Cellpose will run on your CPU instead. This is much slower but more compatible
with different hardware setups.

Note that, particularly when in 3D mode, lack of GPU memory can become a limitation. If a model crashes you may need to
re-start CellProfiler to release GPU memory. Resizing large images prior to running them through the model can free up
GPU memory.
""",
        )

        self.model_directory = Directory(
            "Location of the pre-trained model file",
            doc=f"""\
*(Used only when using a custom pre-trained model)*
Select the location of the pre-trained CellPose model file that will be used for detection.""",
        )

        def get_directory_fn():
            """Get the directory for the rules file name"""
            return self.model_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.model_directory.get_parts_from_path(path)

            self.model_directory.join_parts(dir_choice, custom_path)

        self.model_file_name = Filename(
            "Pre-trained model file name",
            "cyto_0",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn,
            doc=f"""\
*(Used only when using a custom pre-trained model)*
This file can be generated by training a custom model withing the CellPose GUI or command line applications.""",
        )

        self.gpu_test = DoSomething(
            "",
            "Test GPU",
            self.do_check_gpu,
            doc=f"""\
Press this button to check whether a GPU is correctly configured.

If you have a dedicated GPU, a failed test usually means that either your GPU does not support deep learning or the
required dependencies are not installed.
If you have multiple GPUs on your system, this button will only test the first one.
""",
        )

        self.sliding_window_size_0 = Integer(
            text="First dimension of the MONAI SlidingWindowInferer roi_size parameter",
            value=256,
            minval=128,
            doc="""\
First dimension size of the sliding window roi_size parameter, default to 256
""",
        )

        self.sliding_window_size_1 = Integer(
            text="Second dimension of the MONAI SlidingWindowInferer roi_size parameter",
            value=256,
            minval=128,
            doc="""\
Second dimension size of the sliding window roi_size parameter, default to 256
""",
        )

        self.multigpu_infer = Binary(
            text="Whether using multi-GPU to perform the inference",
            value=False,
            doc="""\
Whether using multiple GPUs to perform the inference
"""
        )

    def settings(self):
        return [
            self.x_name,
            self.docker_or_python,
            self.y_name,
            self.use_gpu,
            self.model_directory,
            self.model_file_name,
            self.sliding_window_size_0,
            self.sliding_window_size_1,
            self.multigpu_infer,
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.docker_or_python,
            self.y_name,
            self.use_gpu,
            self.model_directory,
            self.model_file_name,
            self.sliding_window_size_0,
            self.sliding_window_size_1,
            self.multigpu_infer,
        ]

    def verify_bundle(self, pipeline):
        """If using custom model, validate the model file opens and works"""
        pass
    
    @classmethod
    def download_vista2d(cls, save_path):
        if os.path.exists(save_path):
            return
        
        os.makedirs(save_path)
        monai.bundle.download(name=cls.VISTA2D_BUNDLE_NAME, version=cls.VISTA2D_BUNDLE_VERSION, bundle_dir=save_path)


    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value
        images = workspace.image_set
        x = images.get_image(x_name)
        dimensions = x.dimensions
        x_data = x.pixel_data
        self.download_vista2d(BUNDLE_PATH)
        bundle_path = os.path.join(BUNDLE_PATH, self.VISTA2D_BUNDLE_NAME)
        

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_img_dir = os.path.join(temp_dir, "img")
            temp_img_path = os.path.join(temp_img_dir, x_name+".tiff")
            temp_mask_dir = os.path.join(temp_dir, "mask")
            temp_label_path = os.path.join(temp_mask_dir, y_name+".tif")
            skimage.io.imsave(temp_img_path, x_data)
            if self.docker_or_python.value == "Python":
                if self.multigpu_infer:
                    pass
                else:
                    cmd = f"""cd {bundle_path};
                    python -m monai.bundle run_workflow "scripts.workflow.VistaCell"\
                           --config_file configs/hyper_parameters.yaml\
                           --mode infer --pretrained_ckpt_name vista2d_v1.pt
                    """
                
            elif self.docker_or_python.value == "Docker":
                # Define how to call docker
                docker_path = "docker" if sys.platform.lower().startswith("win") else "/usr/local/bin/docker"
                # Create a UUID for this run
                unique_name = str(uuid.uuid4())
                # Directory that will be used to pass images to the docker container
                temp_dir = os.path.join(get_default_output_directory(), ".cellprofiler_temp", unique_name)
                temp_img_dir = os.path.join(temp_dir, "img")
                
                os.makedirs(temp_dir, exist_ok=True)
                os.makedirs(temp_img_dir, exist_ok=True)

                
                if self.mode.value == "custom":
                    model_file = self.model_file_name.value
                    model_directory = self.model_directory.get_absolute_path()
                    model_path = os.path.join(model_directory, model_file)
                    temp_model_dir = os.path.join(temp_dir, "model")

                    os.makedirs(temp_model_dir, exist_ok=True)
                    # Copy the model
                    shutil.copy(model_path, os.path.join(temp_model_dir, model_file))

                # Save the image to the Docker mounted directory
                skimage.io.imsave(temp_img_path, x_data)

                cmd = f"""
                {docker_path} run --rm -v {temp_dir}:/data
                {self.docker_image.value}
                {'--gpus all' if self.use_gpu.value else ''}
                cellpose
                --dir /data/img
                {'--pretrained_model ' + self.mode.value if self.mode.value != 'custom' else '--pretrained_model /data/model/' + model_file}
                --chan {channels[0]}
                --chan2 {channels[1]}
                --diameter {diam}
                {'--net_avg' if self.use_averaging.value else ''}
                {'--do_3D' if self.do_3D.value else ''}
                --anisotropy {anisotropy}
                --flow_threshold {self.flow_threshold.value}
                --cellprob_threshold {self.cellprob_threshold.value}
                --stitch_threshold {self.stitch_threshold.value}
                --min_size {self.min_size.value}
                {'--invert' if self.invert.value else ''}
                {'--exclude_on_edges' if self.remove_edge_masks.value else ''}
                --verbose
                """

                try:
                    subprocess.run(cmd.split(), text=True)
                    cellpose_output = numpy.load(os.path.join(temp_img_dir, unique_name + "_seg.npy"), allow_pickle=True).item()

                    y_data = cellpose_output["masks"]
                    flows = cellpose_output["flows"]
                finally:      
                    # Delete the temporary files
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        LOGGER.error("Unable to delete temporary directory, files may be in use by another program.")
                        LOGGER.error("Temp folder is subfolder {tempdir} in your Default Output Folder.\nYou may need to remove it manually.")


        y = Objects()
        y.segmented = y_data
        y.parent_image = x.parent_image
        objects = workspace.object_set
        objects.add_objects(y, y_name)

        if self.save_probabilities.value:
            # Flows come out sized relative to CellPose's inbuilt model size.
            # We need to slightly resize to match the original image.
            size_corrected = skimage.transform.resize(flows[2], y_data.shape)
            prob_image = Image(
                size_corrected,
                parent_image=x.parent_image,
                convert=False,
                dimensions=len(size_corrected.shape),
            )

            workspace.image_set.add(self.probabilities_name.value, prob_image)

            if self.show_window:
                workspace.display_data.probabilities = size_corrected

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data
            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        if self.save_probabilities.value:
            layout = (2, 2)
        else:
            layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.x_data,
            title="Input Image",
            x=0,
            y=0,
        )

        figure.subplot_imshow_labels(
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            x=1,
            y=0,
        )
        if self.save_probabilities.value:
            figure.subplot_imshow(
                colormap="gray",
                image=workspace.display_data.probabilities,
                sharexy=figure.subplot(0, 0),
                title=self.probabilities_name.value,
                x=0,
                y=1,
            )

    def do_check_gpu(self):
        import importlib.util
        torch_installed = importlib.util.find_spec('torch') is not None
        self.cellpose_ver = importlib.metadata.version('cellpose')
        #if the old version of cellpose <2.0, then use istorch kwarg
        if float(self.cellpose_ver[0:3]) >= 0.7 and int(self.cellpose_ver[0])<2:
            GPU_works = core.use_gpu(istorch=torch_installed)
        else:  # if new version of cellpose, use use_torch kwarg
            GPU_works = core.use_gpu(use_torch=torch_installed)
        if GPU_works:
            message = "GPU appears to be working correctly!"
        else:
            message = (
                "GPU test failed. There may be something wrong with your configuration."
            )
        import wx

        wx.MessageBox(message, caption="GPU Test")

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            setting_values = setting_values + ["0.4", "0.0"]
            variable_revision_number = 2
        if variable_revision_number == 2:
            setting_values = setting_values + ["0.0", False, "15", "1.0", False, False]
            variable_revision_number = 3
        if variable_revision_number == 3:
            setting_values = [setting_values[0]] + ["Python",CELLPOSE_DOCKER_IMAGE_WITH_PRETRAINED] + setting_values[1:]
            variable_revision_number = 4
        if variable_revision_number == 4:
            setting_values = [setting_values[0]] + ['No'] + setting_values[1:]
            variable_revision_number = 5
        return setting_values, variable_revision_number
    

