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
import logging
import os
import platform
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from distutils.util import strtobool
from math import ceil
from typing import Any, Callable, Dict, Optional, Sequence, Union

import openslide
import requests
import schedule
from dicomweb_client.session_utils import create_session_from_user_pass
from monai.apps import download_and_extract, download_url, load_from_mmar
from monai.data import partition_dataset
from timeloop import Timeloop

from monailabel.config import settings
from monailabel.datastore.dicom import DICOMwebClientX, DICOMWebDatastore
from monailabel.datastore.local import LocalDatastore
from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.interfaces.tasks.batch_infer import BatchInferImageType, BatchInferTask
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.infer.deepgrow_2d import InferDeepgrow2D
from monailabel.tasks.infer.deepgrow_3d import InferDeepgrow3D
from monailabel.tasks.infer.deepgrow_pipeline import InferDeepgrowPipeline
from monailabel.utils.async_tasks.task import AsyncTask
from monailabel.utils.others.pathology import create_asap_annotations_xml, create_dsa_annotations_json
from monailabel.utils.sessions import Sessions

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
        labels: Union[Optional[Sequence[str]], Optional[Dict[Any, Any]]] = None,
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

        if strtobool(conf.get("download_tools", "true")):
            self._download_tools()
        self._server_mode = strtobool(conf.get("server_mode", "false"))
        self._auto_update_scoring = strtobool(conf.get("auto_update_scoring", "true"))
        self._sessions = self._load_sessions(strtobool(conf.get("sessions", "true")))

        self._infers_threadpool = (
            None
            if settings.MONAI_LABEL_INFER_CONCURRENCY < 0
            else ThreadPoolExecutor(max_workers=settings.MONAI_LABEL_INFER_CONCURRENCY, thread_name_prefix="INFER")
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {}

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {}

    def init_strategies(self) -> Dict[str, Strategy]:
        return {"random": Random()}

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        return {}

    def init_batch_infer(self) -> Callable:
        return BatchInferTask()

    def init_datastore(self) -> Datastore:
        logger.info(f"Init Datastore for: {self.studies}")
        if self.studies.startswith("http://") or self.studies.startswith("https://"):
            self.studies = self.studies.rstrip("/").strip()
            logger.info(f"Using DICOM WEB: {self.studies}")

            dw_session = None
            if settings.MONAI_LABEL_DICOMWEB_USERNAME and settings.MONAI_LABEL_DICOMWEB_PASSWORD:
                dw_session = create_session_from_user_pass(
                    settings.MONAI_LABEL_DICOMWEB_USERNAME, settings.MONAI_LABEL_DICOMWEB_PASSWORD
                )

            dw_client = DICOMwebClientX(
                url=self.studies,
                session=dw_session,
                qido_url_prefix=settings.MONAI_LABEL_QIDO_PREFIX,
                wado_url_prefix=settings.MONAI_LABEL_WADO_PREFIX,
                stow_url_prefix=settings.MONAI_LABEL_STOW_PREFIX,
            )

            cache_path = settings.MONAI_LABEL_DICOMWEB_CACHE_PATH
            cache_path = cache_path.strip() if cache_path else ""
            fetch_by_frame = settings.MONAI_LABEL_DICOMWEB_FETCH_BY_FRAME
            return (
                DICOMWebDatastore(dw_client, cache_path, fetch_by_frame=fetch_by_frame)
                if cache_path
                else DICOMWebDatastore(dw_client, fetch_by_frame=fetch_by_frame)
            )

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
            merged = []
            for labels in [v.get("labels", []) for v in meta["models"].values()]:
                if labels and isinstance(labels, dict):
                    labels = [k for k, _ in sorted(labels.items(), key=lambda item: item[1])]  # type: ignore
                for l in labels:
                    if l not in merged:
                        merged.append(l)
            meta["labels"] = merged

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
        model = request.get("model")
        if not model:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                "Model is not provided for Inference Task",
            )

        task = self._infers.get(model)
        if not task:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                f"Inference Task is not Initialized. There is no model '{model}' available",
            )

        request = copy.deepcopy(request)
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

        logger.debug(f"Image => {request['image']}")
        if self._infers_threadpool:

            def run_infer_in_thread(t, r):
                return t(r)

            f = self._infers_threadpool.submit(run_infer_in_thread, t=task, r=request)
            result_file_name, result_json = f.result(request.get("timeout", settings.MONAI_LABEL_INFER_TIMEOUT))
        else:
            result_file_name, result_json = task(request)

        label_id = None
        if result_file_name and os.path.exists(result_file_name):
            tag = request.get("label_tag", DefaultLabelTag.ORIGINAL)
            save_label = request.get("save_label", False)
            if save_label:
                label_id = datastore.save_label(image_id, result_file_name, tag, dict())
            else:
                label_id = result_file_name

        return {"label": label_id, "tag": DefaultLabelTag.ORIGINAL, "file": result_file_name, "params": result_json}

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
        if not method:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                "Method is not provided for Scoring Task",
            )

        task = self._scoring_methods.get(method)
        if not task:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                f"Scoring Task is not Initialized. There is no such scoring method '{method}' available",
            )

        request = copy.deepcopy(request)
        return task(copy.deepcopy(request), datastore if datastore else self.datastore())

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
                        "model": "mytrain",
                        "device": "cuda"
                        "max_epochs": 1,
                    }

        Returns:
            JSON containing train stats
        """
        model = request.get("model")
        if not model:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                "Model is not provided for Training Task",
            )

        task = self._trainers.get(model)
        if not task:
            raise MONAILabelException(
                MONAILabelError.INVALID_INPUT,
                f"Train Task is not Initialized. There is no model '{model}' available",
            )

        request = copy.deepcopy(request)
        result = task(request, self.datastore())

        # Run all scoring methods
        if self._auto_update_scoring:
            self.async_scoring(None)
        return result

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

        # Run all scoring methods
        if self._auto_update_scoring:
            self.async_scoring(None)

        return {
            "id": image_id,
            "path": image_path,
        }

    def on_init_complete(self):
        logger.info("App Init - completed")

        # Run all scoring methods
        if self._auto_update_scoring:
            self.async_scoring(None)

        # Run Cleanup Jobs
        def cleanup_sessions(instance):
            instance.cleanup_sessions()

        cleanup_sessions(self)
        time_loop = Timeloop()
        schedule.every(5).minutes.do(cleanup_sessions, self)

        @time_loop.job(interval=timedelta(seconds=30))
        def run_scheduler():
            schedule.run_pending()

        time_loop.start(block=False)

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
        if not method and not self._scoring_methods:
            return {}

        methods = [method] if method else list(self._scoring_methods.keys())
        result = {}
        for m in methods:
            if self._server_mode:
                request = {"method": m}
                request.update(params[m] if params and params.get(m) else {})
                res, _ = AsyncTask.run("scoring", request=request, params=params, enqueue=True)
                result[m] = res
            else:
                url = f"/scoring/{m}"
                p = params[m] if params and params.get(m) else None
                result[m] = self._local_request(url, p, "Scoring")
        return result[method] if method else result

    def async_training(self, model, params=None, enqueue=False):
        if not model and not self._trainers:
            return {}

        models = [model] if model else list(self._trainers.keys())
        enqueue = True if model > 1 else enqueue
        result = {}
        for m in models:
            if self._server_mode:
                request = {"model": m}
                request.update(params[m] if params and params.get(m) else {})
                res, _ = AsyncTask.run("train", request=request, params=params, enqueue=enqueue)
                result[m] = res
            else:
                url = f"/train/{model}?enqueue={enqueue}"
                p = params[m] if params and params.get(m) else None
                result[m] = self._local_request(url, p, "Training")
        return result[model] if model else result

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
        logger.debug(f"Existing Tools: {existing}")

        if len(existing) in [len(dcmqi_tools), len(dcmqi_tools) // 2]:
            logger.debug("No need to download dcmqi tools")
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

    def _load_sessions(self, load=False):
        if not load:
            return None
        return Sessions(settings.MONAI_LABEL_SESSION_PATH, settings.MONAI_LABEL_SESSION_EXPIRY)

    def cleanup_sessions(self):
        if not self._sessions:
            return
        count = self._sessions.remove_expired()
        logger.debug("Total sessions cleaned up: {}".format(count))

    def sessions(self):
        return self._sessions

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
        deepgrow_2d = load_from_mmar("clara_pt_deepgrow_2d_annotation", model_dir)
        deepgrow_3d = load_from_mmar("clara_pt_deepgrow_3d_annotation", model_dir)

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

    def infer_wsi(self, request, datastore=None):
        image = request["image"]

        # Possibly direct image (numpy)
        if not isinstance(image, str):
            res = self.infer(request, datastore)
            logger.info(f"Latencies: {res.get('params', {}).get('latencies')}")
            return res

        request = copy.deepcopy(request)
        if not os.path.exists(image):
            datastore = datastore if datastore else self.datastore()
            image = datastore.get_image_uri(request["image"])

        start = time.time()
        logger.debug(f"Input WSI Image: {image}")
        infer_tasks = self._create_infer_wsi_tasks(request, image)
        logger.debug(f"Total WSI Tasks: {len(infer_tasks)}")

        res_json = {"tasks": {}}
        for t in infer_tasks:
            tid = t["id"]
            res = self._run_infer_wsi_task(t)
            res_json["tasks"][tid] = res
            logger.info(f"{tid} => Completed: {len(res_json)} / {len(infer_tasks)}; Latencies: {res.get('latencies')}")

        latency_total = time.time() - start
        logger.debug("WSI Infer Time Taken: {:.4f}".format(latency_total))

        res_json.update(
            {
                "latencies": {
                    "total": round(latency_total, 2),
                },
            }
        )

        res_file = None
        output = request.get("output", "asap")
        logger.debug(f"+++ WSI Inference Output Type: {output}")

        loglevel = request.get("logging", "INFO").upper()
        if output == "asap":
            logger.debug("+++ Generating ASAP XML Annotation")
            res_file = create_asap_annotations_xml(res_json, color_map=request.get("color_map"), loglevel=loglevel)
        elif output == "dsa":
            logger.debug("+++ Generating DSA JSON Annotation")
            model = request.get("model")
            task = self._infers.get(model)
            res_file = create_dsa_annotations_json(
                res_json,
                name=f"MONAILabel - {model}",
                description=task.description,
                color_map=request.get("color_map"),
                loglevel=loglevel,
            )
        else:
            logger.debug("+++ Return Default JSON Annotation")

        if len(infer_tasks) > 1:
            logger.info("Total Time Taken: {:.4f}; Total Infer Time: {:.4f}".format(time.time() - start, latency_total))
        return {"file": res_file, "params": res_json}

    def _run_infer_wsi_task(self, task):
        tid = task["id"]
        (row, col, tx, ty, tw, th) = task["coords"]
        logger.debug(f"{tid} => Patch/Slide ({row}, {col}) => Location: ({tx}, {ty}); Size: {tw} x {th}")

        req = copy.deepcopy(task)
        req["result_write_to_file"] = False
        req["logging"] = req.get("logging", "WARNING")

        res = self.infer(req)
        return res.get("params", {})

    def _create_infer_wsi_tasks(self, request, image):
        tile_size = request.get("tile_size", (2048, 2048))
        tile_size = [int(p) for p in tile_size]

        # TODO:: Auto-Detect based on WSI dimensions instead of 3000
        min_poly_area = request.get("min_poly_area", 3000)

        location = request.get("location", [0, 0])
        size = request.get("size", [0, 0])
        bbox = [[location[0], location[1]], [location[0] + size[0], location[1] + size[1]]]
        bbox = bbox if bbox and sum(bbox[0]) + sum(bbox[1]) > 0 else None
        level = request.get("level", 0)

        with openslide.OpenSlide(image) as slide:
            w, h = slide.dimensions
        logger.debug(f"Input WSI Image Dimensions: ({w} x {h}); Tile Size: {tile_size}")

        x, y = 0, 0
        if bbox:
            x, y = int(bbox[0][0]), int(bbox[0][1])
            w, h = int(bbox[1][0] - x), int(bbox[1][1] - y)
            logger.debug(f"WSI Region => Location: ({x}, {y}); Dimensions: ({w} x {h})")

        cols = ceil(w / tile_size[0])  # COL
        rows = ceil(h / tile_size[1])  # ROW

        if rows * cols > 1:
            logger.info(f"Total Tiles to infer {rows} x {cols}: {rows * cols}")

        infer_tasks = []
        count = 0
        pw, ph = tile_size[0], tile_size[1]
        for row in range(rows):
            for col in range(cols):
                tx = col * pw + x
                ty = row * ph + y

                tw = min(pw, x + w - tx)
                th = min(ph, y + h - ty)

                task = copy.deepcopy(request)
                task.update(
                    {
                        "id": count,
                        "image": image,
                        "tile_size": tile_size,
                        "min_poly_area": min_poly_area,
                        "coords": (row, col, tx, ty, tw, th),
                        "location": (tx, ty),
                        "level": level,
                        "size": (tw, th),
                    }
                )
                infer_tasks.append(task)
                count += 1
        return infer_tasks
