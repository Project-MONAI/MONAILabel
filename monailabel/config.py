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
