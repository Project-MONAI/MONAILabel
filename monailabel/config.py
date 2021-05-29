from typing import List

from pydantic import AnyHttpUrl, BaseSettings


class Settings(BaseSettings):
    API_STR: str = ""
    PROJECT_NAME: str = "MONAILabel"

    APP_DIR: str = ""
    STUDIES: str = ""

    DATASTORE_AUTO_RELOAD: bool = True
    DATASTORE_IMAGE_EXT: List[str] = ["*.nii.gz", "*.nii"]
    DATASTORE_LABEL_EXT: List[str] = ["*.nii.gz", "*.nii"]

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
