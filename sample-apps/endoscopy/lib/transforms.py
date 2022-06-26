# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Optional

from monai.data import ImageReader
from monai.transforms import LoadImaged

logger = logging.getLogger(__name__)


class LoadImageExd(LoadImaged):
    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        objs = {}
        for key in self.keys:
            if not isinstance(d[key], str):
                objs[key] = d[key]
                d.pop(key)
                continue  # Support direct image in np (pass only transform)

        if len(objs) < len(d):
            d = super().__call__(data, reader)
            d.update(objs)
        return d
