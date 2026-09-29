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

"""Fixed discovery and inference destinations; catalogs cannot redirect credentials."""

from dataclasses import dataclass
from typing import Literal

from monailabel.providers.vision import VisionProvider

ServiceId = Literal["nvidia", "openai", "anthropic", "gemini"]


@dataclass(frozen=True)
class HostedService:
    id: ServiceId
    name: str
    token_env: str
    catalog_url: str
    inference_url: str
    provider: VisionProvider
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    responses_url: str | None = None


SERVICES: dict[ServiceId, HostedService] = {
    "nvidia": HostedService(
        "nvidia",
        "NVIDIA gateway",
        "NV_INFERENCE_API_KEY",
        "https://inference-api.nvidia.com/v1/models",
        "https://inference-api.nvidia.com/v1/chat/completions",
        "openai-chat-polygons",
        responses_url="https://inference-api.nvidia.com/v1/responses",
    ),
    "openai": HostedService(
        "openai",
        "OpenAI",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1/models",
        "https://api.openai.com/v1/responses",
        "openai-polygons",
    ),
    "anthropic": HostedService(
        "anthropic",
        "Anthropic (Claude)",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/models",
        "https://api.anthropic.com/v1/messages",
        "anthropic-polygons",
    ),
    "gemini": HostedService(
        "gemini",
        "Google (Gemini)",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/models",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "openai-chat-polygons",
        "max_tokens",
    ),
}
