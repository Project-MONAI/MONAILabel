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

from typing import Any, Dict, List

from pydantic.main import BaseModel


class DICOMObjectModel(BaseModel):
    patient_id: str
    study_id: str
    series_id: str
    local_path: str = ""
    info: Dict[str, Any] = {}
    memory_cache: Dict[str, Any] = {}
    tag: str = ""
    related_labels_keys: List[str] = []


class DICOMWebDatastoreModel(BaseModel):
    url: str
    description: str
    objects: Dict[str, DICOMObjectModel] = {}
