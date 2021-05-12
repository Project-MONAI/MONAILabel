import logging
import os
from abc import abstractmethod

import yaml

from monailabel.interfaces.datastore import Datastore, LabelTag
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.utils.activelearning import Random
from monailabel.utils.datastore import LocalDatastore

logger = logging.getLogger(__name__)


class MONAILabelApp:
    def __init__(
        self,
        app_dir,
        studies,
        infers=None,
        strategies=None,
    ):
        """
        Base Class for Any MONAI Label App

        :param app_dir: path for your App directory
        :param studies: path for studies/datalist
        :param infers: Dictionary of infer engines
        :param strategies: List of ActiveLearning strategies to get next sample

        """
        self.app_dir = app_dir
        self.studies = studies
        self.infers = dict() if infers is None else infers
        self.strategies = {"random", Random()} if strategies is None else strategies

        self._datastore: Datastore = LocalDatastore(studies)

    def info(self):
        """
        Provide basic information about APP.  This information is passed to client.
        Default implementation is to pass the contents of info.yaml present in APP_DIR
        """
        file = os.path.join(self.app_dir, "info.yaml")
        if not os.path.exists(file):
            raise MONAILabelException(MONAILabelError.APP_ERROR, "info.yaml NOT Found in the APP Folder")

        with open(file, "r") as fc:
            meta = yaml.full_load(fc)

        models = dict()
        for name, infer in self.infers.items():
            if infer.is_valid():
                models[name] = infer.info()
        meta["models"] = models

        strategies = dict()
        for name, strategy in self.strategies.items():
            strategies[name] = strategy.info()
        meta["strategies"] = strategies

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
                        "save_label": "true/false",
                        "params": {},
                    }

        Raises:
            MONAILabelException: When ``model`` is not found

        Returns:
            JSON containing `label` and `params`
        """
        model_name = request.get("model")
        model_name = model_name if model_name else "model"

        task = self.infers.get(model_name)
        if task is None:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "Inference Task is not Initialized. There is no pre-trained model available",
            )

        image_id = request["image"]
        request["image"] = self._datastore.get_image_uri(request["image"])
        result_file_name, result_json = task(request)

        if request.get("save_label", True):
            self.datastore().save_label(image_id, result_file_name, LabelTag.ORIGINAL)

        return {"label": result_file_name, "params": result_json}

    def datastore(self) -> Datastore:
        return self._datastore

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
                        "strategy": "random"
                    }

        Returns:
            JSON containing next image info that is selected for labeling
        """
        strategy = request.get("strategy")
        strategy = strategy if strategy else "random"

        task = self.strategies.get(strategy)
        if task is None:
            raise MONAILabelException(
                MONAILabelError.APP_INIT_ERROR,
                f"ActiveLearning Task is not Initialized. There is no such strategy '{strategy}' available",
            )

        image_id = task(request, self.datastore())
        image_path = self._datastore.get_image_uri(image_id)
        return {
            "id": image_id,
            "path": image_path,
        }

    def save_label(self, request):
        """
        Saving New Label.  You can extend this has callback handler to run calibrations etc. over Active learning models

        Args:
            request: JSON object which contains Label and Image details

                For example::

                    {
                        "image": "file://xyz.com",
                        "label": "file://label_xyz.com",
                        "segments" ["spleen"],
                        "params": {},
                    }

        Returns:
            JSON containing next image and label info
        """

        label_id = self.datastore().save_label(request["image"], request["label"], LabelTag.FINAL)

        return {
            "image": request.get("image"),
            "label": label_id,
        }
