from abc import abstractmethod
from typing import Any

from monai.transforms.transform import Transform


class ProtoSegmentation(Transform):
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        raise NotImplementedError
