from abc import abstractmethod
from typing import Any

from monai.transforms.transform import Transform


class Seeding(Transform):

    @abstractmethod
    def __call__(self, data: Any, interest_definition: Any = None) -> Any:
        raise NotImplementedError
