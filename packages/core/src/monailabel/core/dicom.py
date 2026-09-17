"""DICOM source connections, searches and import selections."""

from datetime import date
from typing import Literal, Self

from pydantic import Field, model_validator

from monailabel.core.models import Contract, Record


class DicomConnection(Record):
    project_id: str
    name: str
    url: str
    authentication: Literal["none", "basic", "bearer"] = "none"
    credential_id: str | None = None


class DicomFilters(Contract):
    modality: str = Field(default="", max_length=16, pattern=r"^[A-Z0-9]*$")
    date_from: date | None = None
    date_to: date | None = None
    patient_id: str = Field(default="", max_length=128)
    patient_name: str = Field(default="", max_length=128)
    study_description: str = Field(default="", max_length=128)
    series_description: str = Field(default="", max_length=128)
    accession_number: str = Field(default="", max_length=128)
    study_uid: str = Field(default="", max_length=64, pattern=r"^(?:[0-9]+(?:\.[0-9]+)*)?$")
    series_uid: str = Field(default="", max_length=64, pattern=r"^(?:[0-9]+(?:\.[0-9]+)*)?$")
    import_status: Literal["new", "all", "imported"] = "new"

    @model_validator(mode="after")
    def date_order(self) -> Self:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("The start date must be on or before the end date.")
        return self


class DicomSeriesRef(Contract):
    study_uid: str = Field(min_length=1, max_length=64, pattern=r"^[0-9]+(?:\.[0-9]+)*$")
    series_uid: str = Field(min_length=1, max_length=64, pattern=r"^[0-9]+(?:\.[0-9]+)*$")


class RemoteDicomSeries(DicomSeriesRef):
    patient_id: str = ""
    patient_name: str = ""
    study_date: str = ""
    study_description: str = ""
    series_description: str = ""
    accession_number: str = ""
    modality: str = ""
    instance_count: int | None = None
    imported_asset_id: str | None = None
    unsupported_reason: str | None = None


class DicomSearch(Contract):
    items: list[RemoteDicomSeries]
    truncated: bool = False
    limit: int = 1000


class DicomImportSelection(Contract):
    series: list[DicomSeriesRef] = Field(min_length=1, max_length=1000)
