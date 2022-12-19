from abc import abstractmethod
from typing import Any

from monai.transforms.transform import Transform


class Mapping(Transform):
    @abstractmethod
    def __call__(self, data: Any, seeding: Any = None) -> Any:
        raise NotImplementedError
