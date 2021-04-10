import logging
import os
import random

from lib import MyInfer, MyTrain
from monailabel.engines.infer import Deepgrow2D, Deepgrow3D, SegmentationSpleen
from monailabel.interface.app import MONAILabelApp

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir):
        super().__init__(
            app_dir=app_dir,
            studies=os.path.join(app_dir, "studies"),
            cache=True,
            infer_models={
                "deepgrow_2d": (Deepgrow2D, "deepgrow_2d.ts"),
                "deepgrow_3d": (Deepgrow3D, "deepgrow_2d.ts"),
                "segmentation_spleen": (SegmentationSpleen, "segmentation_spleen.ts"),
                # Add any more pre-trained models that you are shipping here...
                "model": (MyInfer, "model.ts")
            },
        )

    def info(self):
        return super().info()

    def infer(self, request):
        return super().infer(request)

    ''' 
    # Example Train Request
    request = {
        "device": "cuda"
        "epochs": 1,
        "amp": False,
        "lr": 0.0001,
        "params": {},
    }
    '''

    def train(self, request):
        epochs = request['epochs']
        amp = request.get('amp', False)
        device = request.get('device', 'cuda')
        lr = request.get('lr', 0.0001)

        logger.info(f"Training request: {request}")
        engine = MyTrain(
            output_dir=os.path.join(self.app_dir, "model", "train_0"),
            data_list=os.path.join(self.studies, "dataset.json"),
            data_root=self.studies,
            device=device,
            lr=lr
        )

        stats = engine.run(max_epochs=epochs, amp=amp)
        return stats

    ''' 
    # Example Active Learning Request
    request = {
        "strategy": "random,
        "params": {},
    }
    '''

    def next_sample(self, request):
        logger.info(f"Active Learning request: {request}")
        images_dir = os.path.join(self.studies, "imagesTr")
        images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if
                  os.path.isfile(os.path.join(images_dir, f)) and (f.endswith(".nii.gz") or f.endswith(".nii"))]
        images.sort()

        strategy = request.get("strategy", "random")
        if strategy == "first":
            image = images[0]
        elif strategy == "last":
            image = images[-1]
        else:
            image = random.choice(images)

        logger.info(f"Strategy: {strategy}; Selected Image: {image}")
        return {"image": image}

    ''' 
    # Example Sae Label Request
    request = {
        "image": "file://xyz,
        "label": "file://label_xyz,
        "params": {},
    }
    '''

    def save_label(self, request):
        return {
            "image": request.get("image"),
            "label": request.get("label"),
        }
