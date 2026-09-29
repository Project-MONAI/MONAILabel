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

"""Persistent browser desktop identity, independent of its runtime and HTTP transport."""

from typing import Literal

from monailabel.core.models import Record


class DesktopSession(Record):
    project_id: str
    user_id: str
    asset_id: str
    viewer: Literal["slicer", "qupath"]
    mode: Literal["annotation", "review"]
    credential_id: str
    ended: bool = False
