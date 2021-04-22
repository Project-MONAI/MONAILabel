from monailabel.interface import InferenceEngine, InferType


class MyInfer(InferenceEngine):
    def __init__(self, path):
        super().__init__(
            path=path,
            network=None,
            type=InferType.SEGMENTATION,
            dimension=3,
            description='My Pre-Train model using XYZ Net'
        )

    def pre_transforms(self):
        pass

    def inferer(self):
        pass

    def post_transforms(self):
        pass
