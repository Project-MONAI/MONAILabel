import logging
import os
from abc import abstractmethod

import yaml

from monailabel.interface.exception import MONAILabelException, MONAILabelError
from monailabel.interface.infer import InferenceEngine

logger = logging.getLogger(__name__)


class MONAILabelApp:
    def __init__(self, app_dir, studies, infer_models, cache=True):
        """
        Base Class for Any MONAI Label App

        :param app_dir: path for your App directory
        :param studies: path for studies/datalist
        :param infer_models: list of pre-trained models with infer engines

            For Example::

                # List of (name, engine, model.ts)
                infer_models={
                    "deepgrow_2d": (monailabel.engines.infer.Deepgrow2D, "deepgrow_2d.ts"),
                    "deepgrow_3d": (monailabel.engines.infer.Deepgrow3D, "deepgrow_2d.ts"),
                    "spleen": (monailabel.engines.infer.SegmentationSpleen, "spleen.ts"),
                    "my_model": (myapp.lib.MyInfer, "model.ts")
                }

        :param cache: enable caching for infer engines to accelerate inference for subsequent infer actions

        """
        self.app_dir = app_dir
        self.studies = studies
        self.cached = {} if cache else None
        self.infer_models = infer_models

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

        # Update models dynamically
        valid = set()
        for model_name in self.infer_models:
            _, file = self.infer_models[model_name]
            if os.path.exists(os.path.join(self.app_dir, 'model', file)):
                valid.add(model_name)

        models = {}
        for m in meta.get("models"):
            if m in valid:
                models[m] = meta.get("models")[m]
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

        engine = self.cached.get(model_name) if self.cached is not None else None
        if engine is None:
            if self.infer_models.get(model_name) is None:
                raise MONAILabelException(
                    MONAILabelError.INFERENCE_ERROR,
                    f"Inference Engine for Model '{model_name}' Not Found"
                )

            c, model_file = self.infer_models[model_name]
            model_file = os.path.join(self.app_dir, 'model', model_file)
            if os.path.exists(model_file):
                engine: InferenceEngine = c(model_file)

        if engine is None:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                "Inference Engine is not Initialized. There is no pre-trained model available"
            )

        if self.cached is not None:
            self.cached[model_name] = engine

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

    @abstractmethod
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
        pass

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
        # TODO:: Save label, trigger training (if condition is met)
        return {
            "image": request.get("image"),
            "label": request.get("label"),
        }
