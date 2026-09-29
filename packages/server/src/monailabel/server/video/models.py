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

"""Persistent CVAT identities shared by ingestion, editing and tracking."""

import secrets
from typing import Literal

from pydantic import Field

from monailabel.core.models import Record


class VideoEditor(Record):
    project_id: str
    asset_id: str
    base_revision: int
    mode: Literal["annotation", "review"]
    server_url: str
    task_id: int
    job_id: int | None = None
    ready: bool = False
    label_map: dict[int, int] = Field(default_factory=dict)
    track_ids: dict[int, str] = Field(default_factory=dict)
    submitted_annotation_id: str | None = None


class ManagedCvatRuntime(Record):
    namespace: str = Field(default_factory=lambda: secrets.token_hex(6), pattern=r"^[a-f0-9]{12}$")
