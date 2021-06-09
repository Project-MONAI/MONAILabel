import copy
import logging
import os
import time
from abc import abstractmethod
from typing import Any, Dict

import yaml
from monai.apps import download_url
from monai.data import partition_dataset

from monailabel.config import settings
from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.interfaces.tasks import BatchInferTask
from monailabel.utils.activelearning import Random
from monailabel.utils.datastore import LocalDatastore
from monailabel.utils.infer import InferDeepgrow2D, InferDeepgrow3D
from monailabel.utils.scoring import Dice, Sum

logger = logging.getLogger(__name__)


class MONAILabelApp:
    def __init__(
        self,
        app_dir,
        studies,
        infers=None,
        batch_infer=None,
        strategies=None,
        scoring_methods=None,
        resources=None,
    ):
        """
        Base Class for Any MONAI Label App

        :param app_dir: path for your App directory
        :param studies: path for studies/datalist
        :param infers: Dictionary of infer engines
        :param batch_infer: Batch Infer Task (default: BatchInferTask)
        :param strategies: List of ActiveLearning strategies to get next sample
        :param scoring_methods: List of Scoring methods to available to run over labels
        :param resources: List of Tuples (path, url) to be downloaded when not exists

        """
        self.app_dir = app_dir
        self.studies = studies
        self.infers = dict() if infers is None else infers
        self.strategies = {"random", Random()} if strategies is None else strategies
        self.scoring_methods = {"sum": Sum(), "dice": Dice()} if not scoring_methods else scoring_methods

        self._datastore: Datastore = LocalDatastore(
            studies,
            image_extensions=settings.DATASTORE_IMAGE_EXT,
            label_extensions=settings.DATASTORE_LABEL_EXT,
            auto_reload=settings.DATASTORE_AUTO_RELOAD,
        )

        self._batch_infer = BatchInferTask() if batch_infer is None else batch_infer
        self._download(resources)

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

        meta["models"] = {k: v.info() for k, v in self.infers.items() if v.is_valid()}
        meta["strategies"] = {k: v.info() for k, v in self.strategies.items()}
        meta["scoring"] = {k: v.info() for k, v in self.scoring_methods.items()}

        meta["train_stats"] = self.train_stats()
        meta["datastore"] = self._datastore.status()
        return meta

    def infer(self, request, datastore=None):
        """
        Run Inference for an exiting pre-trained model.

        Args:
            request: JSON object which contains `model`, `image`, `params` and `device`
            datastore: Datastore object.  If None then use default app level datastore to save labels if applicable

                For example::

                    {
                        "device": "cuda"
                        "model": "segmentation_spleen",
                        "image": "file://xyz",
                        "save_label": "true/false",
                        "label_tag": "original"
                    }

        Raises:
            MONAILabelException: When ``model`` is not found

        Returns:
            JSON containing `label` and `params`
        """
        request = copy.deepcopy(request)
        model_name = request.get("model")
        model_name = model_name if model_name else "model"

        task = self.infers.get(model_name)
        if task is None:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "Inference Task is not Initialized. There is no pre-trained model available",
            )

        image_id = request["image"]
        datastore = datastore if datastore else self.datastore()
        request["image"] = datastore.get_image_uri(request["image"])
        result_file_name, result_json = task(request)

        label_id = None
        if result_file_name and os.path.exists(result_file_name):
            tag = request.get("label_tag", DefaultLabelTag.ORIGINAL)
            save_label = request.get("save_label", True)
            if save_label:
                label_id = datastore.save_label(image_id, result_file_name, tag)
                if result_json:
                    datastore.update_label_info(label_id, result_json)

                if os.path.exists(result_file_name):
                    os.unlink(result_file_name)
            else:
                label_id = result_file_name

        return {"label": label_id, "params": result_json}

    def batch_infer(self, request, datastore=None):
        """
        Run batch inference for an existing pre-trained model.

        Args:
            request: JSON object which contains `model`, `params` and `device`
            datastore: Datastore object.  If None then use default app level datastore to fetch the images

                For example::

                    {
                        "device": "cuda"
                        "model": "segmentation_spleen",
                        "images": "unlabeled",
                        "label_tag": "original"
                    }

        Raises:
            MONAILabelException: When ``model`` is not found

        Returns:
            JSON containing `label` and `params`
        """
        return self._batch_infer(request, datastore if datastore else self.datastore(), self.infer)

    def scoring(self, request, datastore=None):
        """
        Run scoring task over labels.

        Args:
            request: JSON object which contains `model`, `params` and `device`
            datastore: Datastore object.  If None then use default app level datastore to fetch the images

                For example::

                    {
                        "device": "cuda"
                        "method": "dice",
                        "y": "final",
                        "y_pred": "original",
                    }

        Raises:
            MONAILabelException: When ``method`` is not found

        Returns:
            JSON containing result of scoring method
        """
        method = request.get("method")
        if not method or not self.scoring_methods.get(method):
            raise MONAILabelException(
                MONAILabelError.APP_INIT_ERROR,
                f"Scoring Task is not Initialized. There is no such scoring method '{method}' available",
            )

        task = self.scoring_methods[method]
        logger.info(f"Running scoring: {method}: {task.info()}")
        return task(request, datastore if datastore else self.datastore())

    def datastore(self) -> Datastore:
        return self._datastore

    @staticmethod
    def partition_datalist(datalist, val_split, shuffle=True):
        if val_split > 0.0:
            return partition_dataset(datalist, ratios=[(1 - val_split), val_split], shuffle=shuffle)
        return datalist, []

    def train_stats(self):
        return {}

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

    def on_save_label(self, image_id, label_id) -> Dict[str, Any]:
        """
        Callback method when label is saved into datastore by a remote client
        """
        # TODO:: Add default recipe here to trigger training etc..
        logger.info(f"New label saved for: {image_id} => {label_id}")

        self.scoring({"method": "dice"})
        return {}

    @staticmethod
    def _download(resources):
        if not resources:
            return

        for resource in resources:
            if not os.path.exists(resource[0]):
                os.makedirs(os.path.dirname(resource[0]), exist_ok=True)
                logger.info(f"Downloading resource: {resource[0]} from {resource[1]}")
                download_url(resource[1], resource[0])
                time.sleep(1)

    def add_deepgrow_infer_tasks(self):
        deepgrow_2d = os.path.join(self.app_dir, "model", "deepgrow_2d.ts")
        deepgrow_3d = os.path.join(self.app_dir, "model", "deepgrow_3d.ts")

        self.infers.update(
            {
                "deepgrow_2d": InferDeepgrow2D(deepgrow_2d),
                "deepgrow_3d": InferDeepgrow3D(deepgrow_3d),
            }
        )

        self._download(
            [
                (deepgrow_2d, "https://www.dropbox.com/s/y9a7tqcos2n4lxf/deepgrow_2d.ts?dl=1"),
                (deepgrow_3d, "https://www.dropbox.com/s/vs6l30mf6t3h3d0/deepgrow_3d.ts?dl=1"),
            ]
        )
