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

from typing import List

from pydantic import AnyHttpUrl, BaseSettings


class Settings(BaseSettings):
    API_STR: str = ""
    PROJECT_NAME: str = "MONAILabel"

    APP_DIR: str = ""
    STUDIES: str = ""
    DICOMWEB_USERNAME: str = ""
    DICOMWEB_PASSWORD: str = ""

    QIDO_PREFIX: str = ""
    WADO_PREFIX: str = ""
    STOW_PREFIX: str = ""

    DATASTORE_AUTO_RELOAD: bool = True
    DATASTORE_IMAGE_EXT: List[str] = ["*.nii.gz", "*.nii"]
    DATASTORE_LABEL_EXT: List[str] = ["*.nii.gz", "*.nii"]

    DICOM_WEB: str = "http://localhost:8042"

    CORS_ORIGINS: List[AnyHttpUrl] = []

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
