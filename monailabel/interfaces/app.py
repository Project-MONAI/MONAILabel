import io
import logging
import os
import pathlib
from abc import abstractmethod

import yaml

from monailabel.interfaces.activelearning import ActiveLearning
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.datastore_local import LocalDatastore
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException

logger = logging.getLogger(__name__)


class MONAILabelApp:
    def __init__(
        self,
        app_dir,
        studies,
        infers=None,
        active_learning: ActiveLearning = ActiveLearning(),
    ):
        """
        Base Class for Any MONAI Label App

        :param app_dir: path for your App directory
        :param studies: path for studies/datalist
        :param infers: Dictionary of infer engines
        :param active_learning: ActiveLearning implementation to get next sample

        """
        self.app_dir = app_dir
        self.studies = studies
        self.infers = dict() if infers is None else infers
        self.active_learning = active_learning
        self._datastore: Datastore = LocalDatastore(studies)

    def info(self):
        """
        Provide basic information about APP.  This information is passed to client.
        Default implementation is to pass the contents of info.yaml present in APP_DIR
        """
        file = os.path.join(self.app_dir, "info.yaml")
        if not os.path.exists(file):
            raise MONAILabelException(
                MONAILabelError.APP_ERROR, "info.yaml NOT Found in the APP Folder"
            )

        with open(file, "r") as fc:
            meta = yaml.full_load(fc)

        models = dict()
        for name, infer in self.infers.items():
            if infer.is_valid():
                models[name] = infer.info()

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
        model_name = request.get("model")
        model_name = model_name if model_name else "model"

        task = self.infers.get(model_name)
        if task is None:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "Inference Task is not Initialized. There is no pre-trained model available",
            )

        result_file_name, result_json = task(request)
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
                        "strategy": "random,
                    }

        Returns:
            JSON containing next image info that is selected for labeling
        """
        image = self.active_learning(request, self.datastore())
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
                        "segments" ["spleen"],
                        "params": {},
                    }

        Returns:
            JSON containing next image and label info
        """

        label = io.BytesIO(open(request["label"], "rb").read())

        img_name = os.path.basename(request["image"]).rsplit(".")[0]
        file_ext = "".join(pathlib.Path(request["label"]).suffixes)
        segments = request.get("segments")
        if not segments:
            segments = self.info().get("labels", [])
        segments = [segments] if isinstance(segments, str) else segments
        segments = "+".join(segments) if len(segments) else "unk"

        label_id = f"label_{segments}_{img_name}{file_ext}"
        label_file = self.datastore().save_label(request["image"], label_id, label)

        return {
            "image": request.get("image"),
            "label": label_file,
        }
