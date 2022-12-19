from abc import abstractmethod
from typing import Any

from monai.transforms.transform import Transform


class InterestDefinition(Transform):
    @abstractmethod
    def __call__(self, data: Any, proto_segmentation: Any = None) -> Any:
        raise NotImplementedError
