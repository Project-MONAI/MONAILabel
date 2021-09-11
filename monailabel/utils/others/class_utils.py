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

import importlib.util
import inspect
import logging

logger = logging.getLogger(__name__)


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.debug(f"module: {module}")
    return module


def get_class_of_subclass(module, class_c):
    for n, o in inspect.getmembers(module):
        logger.debug(f"{n} => {o}")
        if inspect.isclass(o) and n != class_c:
            b = [cls.__name__ for cls in o.__bases__]
            logger.debug(f"Base classes => {b}")
            if class_c in b:
                logger.info(f"Subclass for {class_c} Found: {o}")
                return o
    return None


def get_class_of_subclass_from_file(module_name, file_path, class_c):
    return get_class_of_subclass(module_from_file(module_name, file_path), class_c)
