import logging
from abc import abstractmethod

from monailabel.interfaces.tasks import InferTask, InferType

logger = logging.getLogger(__name__)


class PostProcTask(InferTask):
    """
    PostProc Inference Task Helper
    """

    def __init__(
        self,
        dimension,
        description,
        labels=None,
        input_key="image",
        output_label_key="pred",
        output_json_key="result",
    ):
        """
        :param dimension: Input dimension
        :param description: Description
        :param input_key: Input key for running inference
        :param output_label_key: Output key for storing result/label of inference
        :param output_json_key: Output key for storing result/label of inference
        """
        super().__init__(
            None,
            None,
            InferType.POSTPROCS,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key=input_key,
            output_label_key=output_label_key,
            output_json_key=output_json_key,
        )

    @abstractmethod
    def postproc(self):
        """
        Provide List of postproc transforms

            For Example::

                return [
                    monai.transforms.AddChanneld(keys='pred'),
                    monai.transforms.Activationsd(keys='pred', softmax=True),
                    monai.transforms.AsDiscreted(keys='pred', argmax=True),
                    monai.transforms.SqueezeDimd(keys='pred', dim=0),
                    monai.transforms.ToNumpyd(keys='pred'),
                    monailabel.interface.utils.Restored(keys='pred', ref_image='image'),
                    monailabel.interface.utils.ExtremePointsd(keys='pred', result='result', points='points'),
                    monailabel.interface.utils.BoundingBoxd(keys='pred', result='result', bbox='bbox'),
                ]

        """
        pass

    def run_postprocessor(self, data):
        return self.run_transforms(data, self.postproc(), log_prefix="PROC")

    def pre_transforms(self):
        return []

    def inferer(self):
        pass

    def post_transforms(self):
        return []
