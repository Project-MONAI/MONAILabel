# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import itertools
import logging
import os
import platform
import shutil
import tempfile
import time
from typing import Callable, Dict, Optional, Sequence

import requests
from dicomweb_client import DICOMwebClient
from dicomweb_client.session_utils import create_session_from_user_pass
from monai.apps import download_and_extract, download_url, load_from_mmar
from monai.data import partition_dataset

from monailabel.config import settings
from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.interfaces.tasks.batch_infer import BatchInferImageType, BatchInferTask
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.activelearning.random import Random
from monailabel.utils.async_tasks.task import AsyncTask
from monailabel.utils.datastore.dicom.cache import DICOMWebCache
from monailabel.utils.datastore.local import LocalDatastore
from monailabel.utils.infer.deepgrow_2d import InferDeepgrow2D
from monailabel.utils.infer.deepgrow_3d import InferDeepgrow3D
from monailabel.utils.infer.deepgrow_pipeline import InferDeepgrowPipeline
from monailabel.utils.scoring.dice import Dice
from monailabel.utils.scoring.sum import Sum

logger = logging.getLogger(__name__)


class MONAILabelApp:
    """
    Default Pre-trained Path for downloading models
    """
    PRE_TRAINED_PATH: str = "https://github.com/Project-MONAI/MONAILabel/releases/download/data/"

    def __init__(
        self,
        app_dir: str,
        studies: str,
        conf: Dict[str, str],
        name: str = "",
        description: str = "",
        version: str = "2.0",
        labels: Optional[Sequence[str]] = None,
    ):
        """
        Base Class for Any MONAI Label App

        :param app_dir: path for your App directory
        :param studies: path for studies/datalist
        :param conf: dictionary of key/value pairs provided by user while running the app

        """
        self.app_dir = app_dir
        self.studies = studies
        self.conf = conf if conf else {}

        self.name = name
        self.description = description
        self.version = version
        self.labels = labels

        self._datastore: Datastore = self.init_datastore()

        self._infers = self.init_infers()
        self._trainers = self.init_trainers()
        self._strategies = self.init_strategies()
        self._scoring_methods = self.init_scoring_methods()
        self._batch_infer = self.init_batch_infer()

        self._download_tools()
        self._server_mode = False

    def init_infers(self) -> Dict[str, InferTask]:
        return {}

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {}

    def init_strategies(self) -> Dict[str, Strategy]:
        return {"random": Random()}

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        return {"sum": Sum(), "dice": Dice()}

    def init_batch_infer(self) -> Callable:
        return BatchInferTask()

    def init_datastore(self) -> Datastore:
        logger.info(f"Init Datastore for: {self.studies}")
        if self.studies.startswith("http://") or self.studies.startswith("https://"):
            dw_session = None
            if settings.MONAI_LABEL_DICOMWEB_USERNAME and settings.MONAI_LABEL_DICOMWEB_PASSWORD:
                dw_session = create_session_from_user_pass(
                    settings.MONAI_LABEL_DICOMWEB_USERNAME, settings.MONAI_LABEL_DICOMWEB_PASSWORD
                )

            dw_client = DICOMwebClient(
                url=self.studies,
                session=dw_session,
                qido_url_prefix=settings.MONAI_LABEL_QIDO_PREFIX,
                wado_url_prefix=settings.MONAI_LABEL_WADO_PREFIX,
                stow_url_prefix=settings.MONAI_LABEL_STOW_PREFIX,
            )
            return DICOMWebCache(dw_client)

        return LocalDatastore(
            self.studies,
            extensions=settings.MONAI_LABEL_DATASTORE_FILE_EXT,
            auto_reload=settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD,
        )

    def info(self):
        """
        Provide basic information about APP.  This information is passed to client.
        """
        meta = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "labels": self.labels,
            "models": {k: v.info() for k, v in self._infers.items() if v.is_valid()},
            "trainers": {k: v.info() for k, v in self._trainers.items()},
            "strategies": {k: v.info() for k, v in self._strategies.items()},
            "scoring": {k: v.info() for k, v in self._scoring_methods.items()},
            "train_stats": {k: v.stats() for k, v in self._trainers.items()},
            "datastore": self._datastore.status(),
        }

        # If labels are not provided, aggregate from all individual infers
        if not self.labels:
            meta["labels"] = list(itertools.chain.from_iterable([v.get("labels", []) for v in meta["models"].values()]))

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

        task = self._infers.get(model_name)
        if task is None:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "Inference Task is not Initialized. There is no pre-trained model available",
            )

        image_id = request["image"]
        datastore = datastore if datastore else self.datastore()
        if os.path.exists(image_id):
            request["save_label"] = False
        else:
            request["image"] = datastore.get_image_uri(request["image"])

        # TODO:: BUG In MONAI? Currently can not load DICOM through ITK Loader
        if os.path.isdir(request["image"]):
            logger.info("Input is a Directory; Consider it as DICOM")
            logger.info(os.listdir(request["image"]))
            request["image"] = [os.path.join(f, request["image"]) for f in os.listdir(request["image"])]

        logger.info(f"Image => {request['image']}")
        result_file_name, result_json = task(request)

        label_id = None
        if result_file_name and os.path.exists(result_file_name):
            tag = request.get("label_tag", DefaultLabelTag.ORIGINAL)
            save_label = request.get("save_label", True)
            if save_label:
                label_id = datastore.save_label(image_id, result_file_name, tag, result_json)
                if os.path.exists(result_file_name):
                    os.unlink(result_file_name)
            else:
                label_id = result_file_name

        return {"label": label_id, "tag": DefaultLabelTag.ORIGINAL, "params": result_json}

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
        if not method or not self._scoring_methods.get(method):
            raise MONAILabelException(
                MONAILabelError.APP_INIT_ERROR,
                f"Scoring Task is not Initialized. There is no such scoring method '{method}' available",
            )

        task = self._scoring_methods[method]
        logger.info(f"Running scoring: {method}: {task.info()}")
        return task(request, datastore if datastore else self.datastore())

    def datastore(self) -> Datastore:
        return self._datastore

    @staticmethod
    def partition_datalist(datalist, val_split, shuffle=True):
        if val_split > 0.0:
            return partition_dataset(datalist, ratios=[(1 - val_split), val_split], shuffle=shuffle)
        return datalist, []

    def train(self, request):
        """
        Run Training.  User APP has to implement this method to run training

        Args:
            request: JSON object which contains train configs that are part APP info

                For example::

                    {
                        "mytrain": {
                            "device": "cuda"
                            "max_epochs": 1,
                            "amp": False,
                            "lr": 0.0001,
                        }
                    }

        Returns:
            JSON containing train stats
        """
        model = request.get("model")
        if model and not self._trainers.get(model):
            raise MONAILabelException(
                MONAILabelError.APP_INIT_ERROR,
                f"Trainer Task is not Initialized. There is no such trainer '{model}' available",
            )

        models = [model] if model else self._trainers.keys()
        results = []
        for m in models:
            task = self._trainers[m]
            req = request.get(m, copy.deepcopy(request))
            logger.info(f"Running training: {m}: {task.info()} => {req}")

            result = task(req, self.datastore())
            results.append(result)
        return results[0] if len(results) == 1 else results

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

        task = self._strategies.get(strategy)
        if task is None:
            raise MONAILabelException(
                MONAILabelError.APP_INIT_ERROR,
                f"ActiveLearning Task is not Initialized. There is no such strategy '{strategy}' available",
            )

        image_id = task(request, self.datastore())
        if not image_id:
            return {}

        image_path = self._datastore.get_image_uri(image_id)
        return {
            "id": image_id,
            "path": image_path,
        }

    def on_init_complete(self):
        logger.info("App Init - completed")

    def on_save_label(self, image_id, label_id):
        """
        Callback method when label is saved into datastore by a remote client
        """
        logger.info(f"New label saved for: {image_id} => {label_id}")

    # TODO :: Allow model files to be monitored and call this method when it is published (during training)
    # def on_model_published(self, model):
    #    pass

    def server_mode(self, mode: bool):
        self._server_mode = mode

    def async_scoring(self, method, params=None):
        if self._server_mode:
            request = {"method": method}
            res, _ = AsyncTask.run("scoring", request=request, params=params)
            return res

        url = f"/scoring/{method}"
        return self._local_request(url, params, "Scoring")

    def async_training(self, model, params=None):
        if self._server_mode:
            res, _ = AsyncTask.run("train", params=params)
            return res

        url = "/train"
        params = {"model": model, model: params} if model else params
        return self._local_request(url, params, "Training")

    def async_batch_infer(self, model, images: BatchInferImageType, params=None):
        if self._server_mode:
            request = {"model": model, "images": images}
            res, _ = AsyncTask.run("batch_infer", request=request, params=params)
            return res

        url = f"/batch/infer/{model}?images={images}"
        return self._local_request(url, params, "Batch Infer")

    def _local_request(self, url, params, action):
        params = params if params else {}
        response = requests.post(f"http://127.0.0.1:{settings.MONAI_LABEL_SERVER_PORT}{url}", json=params)

        if response.status_code != 200:
            logger.error(f"Failed To Trigger {action}: {response.text}")
        return response.json() if response.status_code == 200 else None

    def _download_tools(self):
        target = os.path.join(self.app_dir, "bin")
        os.makedirs(target, exist_ok=True)

        dcmqi_tools = ["segimage2itkimage", "itkimage2segimage", "segimage2itkimage.exe", "itkimage2segimage.exe"]
        existing = [tool for tool in dcmqi_tools if shutil.which(tool) or os.path.exists(os.path.join(target, tool))]

        if len(existing) == len(dcmqi_tools) // 2:
            return

        target_os = "win64.zip" if any(platform.win32_ver()) else "linux.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            download_and_extract(
                url=f"https://github.com/QIICR/dcmqi/releases/download/v1.2.4/dcmqi-1.2.4-{target_os}", output_dir=tmp
            )
            for root, _, files in os.walk(tmp):
                for f in files:
                    if f in dcmqi_tools:
                        shutil.copy(os.path.join(root, f), target)

    @staticmethod
    def download(resources):
        if not resources:
            return

        for resource in resources:
            if not os.path.exists(resource[0]):
                os.makedirs(os.path.dirname(resource[0]), exist_ok=True)
                logger.info(f"Downloading resource: {resource[0]} from {resource[1]}")
                download_url(resource[1], resource[0])
                time.sleep(1)

    @staticmethod
    def deepgrow_infer_tasks(model_dir, pipeline=True):
        """
        Dictionary of Default Infer Tasks for Deepgrow 2D/3D
        """
        deepgrow_2d = load_from_mmar("clara_pt_deepgrow_2d_annotation_1", model_dir)
        deepgrow_3d = load_from_mmar("clara_pt_deepgrow_3d_annotation_1", model_dir)

        infers = {
            "deepgrow_2d": InferDeepgrow2D(None, deepgrow_2d),
            "deepgrow_3d": InferDeepgrow3D(None, deepgrow_3d),
        }
        if pipeline:
            infers["deepgrow_pipeline"] = InferDeepgrowPipeline(
                path=None,
                network=deepgrow_2d,
                model_3d=infers["deepgrow_3d"],
                description="Combines Deepgrow 2D model and 3D deepgrow model",
            )
        return infers
