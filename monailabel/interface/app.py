import logging
import io
import os
import shutil
from abc import abstractmethod

import yaml

from monailabel.interface.activelearning import ActiveLearning
from monailabel.interface.exception import MONAILabelException, MONAILabelError
from monailabel.interface.dataset import Dataset, LocalDataset

logger = logging.getLogger(__name__)


class MONAILabelApp:
    def __init__(self, app_dir, studies, infers=dict(), active_learning: ActiveLearning = ActiveLearning()):
        """
        Base Class for Any MONAI Label App

        :param app_dir: path for your App directory
        :param studies: path for studies/datalist
        :param infers: Dictionary of infer engines
        :param active_learning: ActiveLearning implementation to get next sample

        """
        self.app_dir = app_dir
        self.studies = studies
        self.infers = infers
        self.active_learning = active_learning
        self.dataset: Dataset = LocalDataset(studies)

    def info(self):
        """
        Provide basic information about APP.  This information is passed to client.
        Default implementation is to pass the contents of info.yaml present in APP_DIR
        """
        file = os.path.join(self.app_dir, "info.yaml")
        if not os.path.exists(file):
            raise MONAILabelException(
                MONAILabelError.APP_ERROR,
                "info.yaml NOT Found in the APP Folder"
            )

        with open(file, 'r') as fc:
            meta = yaml.full_load(fc)

        models = dict()
        for name, engine in self.infers.items():
            models[name] = engine.info()

        meta["models"] = models
        return meta

    def infer(self, request):
        """
        Run Inference for an exiting pre-trained model.

        Args:
            request: JSON object which contains `model`, `image`, `params` and `device`

                For example::

                    {
                        "device": "cuda"
                        "model": "segmentation_spleen",
                        "image": "file://xyz",
                        "params": {},
                    }

        Raises:
            MONAILabelException: When ``model`` is not found

        Returns:
            JSON containing `label` and `params`
        """
        model_name = request.get('model')
        model_name = model_name if model_name else 'model'

        engine = self.infers.get(model_name)
        if engine is None:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "Inference Engine is not Initialized. There is no pre-trained model available"
            )

        image = request['image']
        params = request.get('params')
        device = request.get('device', 'cuda')

        result_file_name, result_json = engine.run(image, params, device)
        return {"label": result_file_name, "params": result_json}

    @abstractmethod
    def train(self, request):
        """
        Run Training.  User APP has to implement this method to run training

        Args:
            request: JSON object which contains train configs that are part APP info

                For example::

                    {
                        "device": "cuda"
                        "epochs": 1,
                        "amp": False,
                        "lr": 0.0001,
                        "params": {},
                    }

        Returns:
            JSON containing train stats
        """
        pass

    def next_sample(self, request):
        """
        Run Active Learning selection.  User APP has to implement this method to provide next sample for labelling.

        Args:
            request: JSON object which contains active learning configs that are part APP info

                For example::

                    {
                        "strategy": "random,
                        "params": {},
                    }

        Returns:
            JSON containing next image info that is selected for labeling
        """
        logger.info(f"Active Learning request: {request}")
        images = self.dataset.get_unlabeled_images()

        image = self.active_learning.next(request.get("strategy", "random"), images)
        return {"image": image}

    def save_label(self, request):
        """
        Saving New Label.  You can extend this has callback handler to run calibrations etc. over Active learning models

        Args:
            request: JSON object which contains Label and Image details

                For example::

                    {
                        "image": "file://xyz.com",
                        "label": "file://label_xyz.com",
                        "params": {},
                    }

        Returns:
            JSON containing next image and label info
        """
        
        label = io.BytesIO(open(request['label'], 'rb').read())
        label_file = self.dataset.save_label(request['image'], os.path.basename(request['label']), label)
        
        return {
            "image": request.get("image"),
            "label": label_file,
        }
