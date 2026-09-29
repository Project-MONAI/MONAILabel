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

"""Recipe factories: framework imports happen only when a neural job executes."""

from importlib.util import find_spec

from pydantic import JsonValue, ValidationError

from monailabel.core.errors import DomainError
from monailabel.core.models import RecipeInfo
from monailabel.core.ports import Segmenter, Trainer, VolumeSegmenter, VolumeTrainer
from monailabel.providers.local import GaussianSegmenter, GaussianTrainer
from monailabel.providers.vista3d import REVISION as VISTA_REVISION
from monailabel.providers.vista3d import targets as vista_targets
from monailabel.server.storage import Artifacts
from monailabel.totalsegmentator.catalog import DOCUMENTATION
from monailabel.totalsegmentator.catalog import MODELS as TOTAL_MODELS
from monailabel.totalsegmentator.catalog import targets as total_targets
from monailabel.totalsegmentator.config import TotalConfig


class Recipes:
    def __init__(self, artifacts: Artifacts):
        self.artifacts = artifacts

    def list(self) -> list[RecipeInfo]:
        available = find_spec("monailabel.monai") is not None
        vista_defaults: dict[str, JsonValue] = {}
        unet_defaults: dict[str, JsonValue] = {}
        nnunet_defaults: dict[str, JsonValue] = {}
        if available:
            from monailabel.monai.config import UNetConfig
            from monailabel.monai.nnunet_config import NNUNetConfig
            from monailabel.monai.vista_config import VistaConfig

            vista_defaults = VistaConfig().model_dump(mode="json")
            unet_defaults = UNetConfig().model_dump(mode="json")
            nnunet_defaults = NNUNetConfig(modality="CT").model_dump(
                mode="json", exclude={"modality"}
            )
        return [
            RecipeInfo(
                id="nnunet-v2",
                name="nnU-Net v2 · CT / MRI volumes",
                description="Plan and train a 3D network for your targets "
                "from reviewed CT or MRI volumes.",
                available=available and find_spec("nnunetv2") is not None,
                default_config=nnunet_defaults,
                documentation_url="https://github.com/MIC-DKFZ/nnUNet",
                setup="uv run monailabel",
            ),
            *[
                RecipeInfo(
                    id=provider,
                    name=name,
                    description=f"Fine-tune {name} on reviewed volumes using its 3 mm checkpoint.",
                    available=True,
                    default_config=TotalConfig().model_dump(mode="json"),
                    supported_targets=list(total_targets(provider)),
                    target_class_ids=total_targets(provider),
                    documentation_url=DOCUMENTATION,
                )
                for provider, (name, _) in TOTAL_MODELS.items()
            ],
            RecipeInfo(
                id="vista3d",
                name="VISTA3D CT",
                description="Fine-tune the read-only VISTA3D base on reviewed CT organs.",
                available=available,
                default_config=vista_defaults,
                setup="uv run monailabel-server",
                supported_targets=list(vista_targets()),
                target_class_ids=vista_targets(),
                documentation_url=f"https://huggingface.co/MONAI/vista3d/blob/{VISTA_REVISION}/docs/inference.md",
            ),
            RecipeInfo(
                id="monai-unet",
                name="U-Net · 2D images / 3D volumes",
                description="Multi-class segmentation of reviewed RGB images or scalar volumes.",
                available=available,
                default_config=unet_defaults,
                setup="uv run monailabel-server",
            ),
            RecipeInfo(
                id="pixel-gaussian",
                name="Intensity baseline · demo",
                description="A small Gaussian model for checking the learning workflow.",
                available=True,
                demo_only=True,
            ),
        ]

    def validate(self, recipe: str, config: dict[str, JsonValue]) -> dict[str, JsonValue]:
        info = next((r for r in self.list() if r.id == recipe), None)
        if info is None:
            raise DomainError(f"Unknown training recipe '{recipe}'.")
        if not info.available:
            raise DomainError(
                f"The MONAI runtime is unavailable. Start the server with: {info.setup}"
            )
        if recipe == "pixel-gaussian":
            if config:
                raise DomainError("The demonstration baseline does not accept recipe settings.")
            return {}
        try:
            if recipe == "nnunet-v2":
                from monailabel.monai.nnunet_config import NNUNetConfig

                return NNUNetConfig.model_validate(config).model_dump(mode="json")
            if recipe in TOTAL_MODELS:
                parsed = TotalConfig.model_validate(config)
                if not set(parsed.label_mapping.values()) <= set(total_targets(recipe).values()):
                    raise DomainError("Unsupported TotalSegmentator target IDs for this modality.")
                return parsed.model_dump(mode="json")
            if recipe == "vista3d":
                from monailabel.monai.vista_config import VistaConfig

                return VistaConfig.model_validate(config).model_dump(mode="json")
            from monailabel.monai.config import UNetConfig

            return UNetConfig.model_validate(config).model_dump(mode="json")
        except ValidationError as error:
            raise DomainError(f"Invalid {recipe} settings: {error}") from error

    def trainer(self, recipe: str, config: dict[str, JsonValue]) -> Trainer | VolumeTrainer:
        if recipe == "nnunet-v2":
            from monailabel.monai.nnunet_runtime import NNUNetTrainer

            return NNUNetTrainer(config, self.artifacts)
        if recipe in TOTAL_MODELS:
            from monailabel.totalsegmentator.runtime import TotalTrainer

            return TotalTrainer(recipe, config, self.artifacts)
        if recipe == "pixel-gaussian":
            return GaussianTrainer()
        if recipe == "vista3d":
            from monailabel.monai.vista_runtime import VistaTrainer

            return VistaTrainer(config, self.artifacts)
        from monailabel.monai.runtime import ImageUNetTrainer, UNetTrainer

        if config.get("spatial_dims") == 2:
            return ImageUNetTrainer(config, self.artifacts)
        return UNetTrainer(config, self.artifacts)

    def segmenter(self, recipe: str, state: dict[str, JsonValue]) -> Segmenter | VolumeSegmenter:
        self.validate(recipe, state.get("config", {}) if recipe == "nnunet-v2" else {})  # type: ignore[arg-type]
        if recipe == "nnunet-v2":
            from monailabel.monai.nnunet_runtime import NNUNetSegmenter

            return NNUNetSegmenter(state, self.artifacts)
        if recipe in TOTAL_MODELS:
            from monailabel.totalsegmentator.runtime import TotalSegmenter

            return TotalSegmenter(recipe, state, self.artifacts)
        if recipe == "pixel-gaussian":
            return GaussianSegmenter(state)
        if recipe == "vista3d":
            from monailabel.monai.vista_runtime import VistaSegmenter

            return VistaSegmenter(state, self.artifacts)
        from monailabel.monai.config import UNetConfig
        from monailabel.monai.runtime import ImageUNetSegmenter, UNetSegmenter

        if UNetConfig.model_validate(state["config"]).spatial_dims == 2:
            return ImageUNetSegmenter(state, self.artifacts)
        return UNetSegmenter(state, self.artifacts)
