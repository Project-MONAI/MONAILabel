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
from typing import Dict, List

from pydantic import AnyHttpUrl, BaseSettings


class Settings(BaseSettings):
    MONAI_LABEL_API_STR: str = ""
    MONAI_LABEL_PROJECT_NAME: str = "MONAILabel"

    MONAI_LABEL_APP_DIR: str = ""
    MONAI_LABEL_STUDIES: str = ""
    MONAI_LABEL_APP_CONF: Dict[str, str] = {}

    MONAI_LABEL_TASKS_TRAIN: bool = True
    MONAI_LABEL_TASKS_STRATEGY: bool = True
    MONAI_LABEL_TASKS_SCORING: bool = True
    MONAI_LABEL_TASKS_BATCH_INFER: bool = True

    MONAI_LABEL_DICOMWEB_USERNAME: str = ""
    MONAI_LABEL_DICOMWEB_PASSWORD: str = ""
    MONAI_LABEL_DICOMWEB_CACHE_PATH: str = ""
    MONAI_LABEL_QIDO_PREFIX: str = ""
    MONAI_LABEL_WADO_PREFIX: str = ""
    MONAI_LABEL_STOW_PREFIX: str = ""
    MONAI_LABEL_DICOMWEB_FETCH_BY_FRAME: bool = False

    MONAI_LABEL_DATASTORE_AUTO_RELOAD: bool = True
    MONAI_LABEL_DATASTORE_FILE_EXT: List[str] = ["*.nii.gz", "*.nii", "*.nrrd", "*.jpg", "*.png", "*.tif", "*.svs"]

    MONAI_LABEL_SERVER_PORT: int = 8000
    MONAI_LABEL_CORS_ORIGINS: List[AnyHttpUrl] = []

    MONAI_LABEL_AUTO_UPDATE_SCORING = False

    MONAI_LABEL_SESSIONS: bool = True
    MONAI_LABEL_SESSION_PATH: str = ""
    MONAI_LABEL_SESSION_EXPIRY: int = 3600

    MONAI_LABEL_INFER_CONCURRENCY: int = -1
    MONAI_LABEL_INFER_TIMEOUT: int = 600

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
