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
