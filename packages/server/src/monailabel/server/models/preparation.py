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

"""Cheap, read-only catalog preflight. Providers verify pinned bytes again on use."""

from importlib.util import find_spec

from monailabel.core.errors import DomainError
from monailabel.core.models import ModelRecord
from monailabel.server.storage import Artifacts


def trained_files(model: ModelRecord, artifacts: Artifacts) -> bool:
    if not model.state_key:
        return False
    keys: tuple[str, ...] = ("checkpoint_key",)
    if model.provider == "nnunet-v2":
        keys += ("plans_key", "dataset_key", "fingerprint_key")
    try:
        state = artifacts.json(model.state_key)
        if not isinstance(state, dict):
            return False
        return all(
            isinstance(state.get(key), str) and artifacts.path(str(state[key])).is_file()
            for key in keys
        )
    except (DomainError, OSError, ValueError):
        return False


def prepared(model: ModelRecord, artifacts: Artifacts) -> bool:
    provider = model.provider
    if model.state_key and not artifacts.path(model.state_key).is_file():
        return False
    if provider == "threshold":
        return True
    if provider == "pixel-gaussian":
        return bool(model.state_key)
    if provider == "nninteractive":
        from monailabel.nninteractive.environment import prepared_python
        from monailabel.nninteractive.weights import prepared as interactive_ready

        return prepared_python() is not None and interactive_ready()
    if find_spec("torch") is None:
        return False
    if provider in {"sam2", "medsam2"}:
        from monailabel.sam.weights import prepared as sam_ready

        return find_spec("sam2") is not None and sam_ready(provider)
    if provider == "vista3d":
        from monailabel.monai.vista_weights import prepared as vista_ready

        return find_spec("monai") is not None and (
            vista_ready() if model.read_only else trained_files(model, artifacts)
        )
    if provider in {"totalsegmentator-ct", "totalsegmentator-mr"}:
        from monailabel.totalsegmentator.weights import prepared as total_ready

        return (
            find_spec("nnunetv2") is not None
            and total_ready(provider)
            and (not model.state_key or trained_files(model, artifacts))
        )
    if provider in {"monai-unet", "nnunet-v2"}:
        dependency = "monai" if provider == "monai-unet" else "nnunetv2"
        return trained_files(model, artifacts) and find_spec(dependency) is not None
    return False
