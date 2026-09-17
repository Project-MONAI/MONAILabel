"""Bounded QIDO searches and WADO instance downloads from a DICOMweb endpoint."""

import io
import json
import re
from collections.abc import Callable
from email import policy
from email.parser import BytesParser
from typing import Any
from urllib.parse import urlsplit

import httpx
import pydicom

from monailabel.core.dicom import DicomFilters, DicomSeriesRef, RemoteDicomSeries
from monailabel.core.errors import DomainError

MAX_SERIES = 1000
MAX_INSTANCES = 3000
MAX_BYTES = 256 * 1024**2
INCLUDE_TAGS = "00100010,00100020,00080020,00081030,0008103E,00080050,00201209"


def endpoint(value: str) -> str:
    parts = urlsplit(value.strip())
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise DomainError("Enter an HTTP(S) DICOMweb URL without credentials or query parameters.")
    return value.strip().rstrip("/")


def value(row: dict[str, Any], tag: str) -> str:
    values = row.get(tag, {}).get("Value", [])
    first = values[0] if values else ""
    return str(first.get("Alphabetic", "")) if isinstance(first, dict) else str(first)


def query(filters: DicomFilters) -> dict[str, str]:
    params = {"includefield": INCLUDE_TAGS}
    for field, tag in (
        ("modality", "Modality"),
        ("patient_id", "PatientID"),
        ("patient_name", "PatientName"),
        ("study_description", "StudyDescription"),
        ("series_description", "SeriesDescription"),
        ("accession_number", "AccessionNumber"),
        ("study_uid", "StudyInstanceUID"),
        ("series_uid", "SeriesInstanceUID"),
    ):
        text = str(getattr(filters, field)).strip()
        if text:
            params[tag] = f"*{text}*" if field.endswith(("name", "description")) else text
    if filters.date_from or filters.date_to:
        start = filters.date_from.strftime("%Y%m%d") if filters.date_from else ""
        end = filters.date_to.strftime("%Y%m%d") if filters.date_to else ""
        params["StudyDate"] = f"{start}-{end}"
    return params


class DicomwebClient:
    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        self.url = endpoint(url)
        self.headers = headers or {}
        self.transport = transport

    def get(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
        accept: str = "application/dicom+json",
        limit: int = 8 * 1024**2,
    ) -> tuple[bytes, str]:
        try:
            with (
                httpx.Client(
                    headers=self.headers,
                    transport=self.transport,
                    timeout=30,
                    follow_redirects=False,
                ) as client,
                client.stream(
                    "GET", self.url + path, params=params, headers={"Accept": accept}
                ) as response,
            ):
                if response.status_code in {401, 403}:
                    raise DomainError("The DICOM server rejected the credentials or access.")
                if response.is_redirect:
                    raise DomainError(
                        "Use the final DICOMweb endpoint URL; redirects are not followed."
                    )
                response.raise_for_status()
                body = bytearray()
                for block in response.iter_bytes():
                    if len(body) + len(block) > limit:
                        raise DomainError("The DICOM response exceeds the import size limit.")
                    body.extend(block)
                return bytes(body), response.headers.get("content-type", "")
        except httpx.HTTPError as exc:
            raise DomainError(
                "Could not read the DICOMweb endpoint. Check the URL and server credentials."
            ) from exc

    def rows(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        content, _ = self.get(path, params=params)
        try:
            rows = json.loads(content) if content else []
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise ValueError()
            return rows
        except (ValueError, TypeError) as exc:
            raise DomainError(
                "The endpoint did not return DICOM JSON. Enter its DICOMweb URL."
            ) from exc

    def check(self) -> None:
        self.rows("/series", {"limit": "1"})

    def search(self, filters: DicomFilters) -> tuple[list[RemoteDicomSeries], bool]:
        records: list[RemoteDicomSeries] = []
        seen: set[tuple[str, str]] = set()
        offset = 0
        while len(records) <= MAX_SERIES:
            rows = self.rows("/series", {**query(filters), "limit": "100", "offset": str(offset)})
            if not rows:
                break
            before = len(seen)
            for row in rows:
                try:
                    study, series = value(row, "0020000D"), value(row, "0020000E")
                    count = value(row, "00201209")
                    record = RemoteDicomSeries(
                        study_uid=study,
                        series_uid=series,
                        patient_id=value(row, "00100020"),
                        patient_name=value(row, "00100010"),
                        study_date=value(row, "00080020"),
                        study_description=value(row, "00081030"),
                        series_description=value(row, "0008103E"),
                        accession_number=value(row, "00080050"),
                        modality=value(row, "00080060"),
                        instance_count=int(count) if count else None,
                    )
                except (ValueError, TypeError, AttributeError) as exc:
                    raise DomainError(
                        "The server returned incomplete or invalid DICOM series metadata."
                    ) from exc
                if (study, series) in seen:
                    continue
                seen.add((study, series))
                unsupported = None
                if record.modality not in {"CT", "MR"}:
                    unsupported = "Only regular single-frame CT/MR series can be imported."
                elif (
                    record.instance_count is not None
                    and not 1 <= record.instance_count <= MAX_INSTANCES
                ):
                    unsupported = "Import supports 1–3,000 images per series."
                records.append(record.model_copy(update={"unsupported_reason": unsupported}))
                if len(records) > MAX_SERIES:
                    return records[:MAX_SERIES], True
            if len(seen) == before:
                raise DomainError(
                    "The DICOM server is not advancing search pages. Refine the filters."
                )
            offset += len(rows)
        return records, False

    def instances(self, ref: DicomSeriesRef, progress: Callable[[float], None]) -> list[bytes]:
        path = f"/studies/{ref.study_uid}/series/{ref.series_uid}/instances"
        identifiers: list[str] = []
        while len(identifiers) <= MAX_INSTANCES:
            rows = self.rows(path, {"limit": "100", "offset": str(len(identifiers))})
            if not rows:
                break
            for row in rows:
                uid = value(row, "00080018")
                if len(uid) > 64 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", uid):
                    raise DomainError("The DICOM server returned an invalid instance UID.")
                if uid in identifiers:
                    raise DomainError("The DICOM server returned duplicate instance pages.")
                identifiers.append(uid)
            if len(identifiers) > MAX_INSTANCES:
                break
        if not 1 <= len(identifiers) <= MAX_INSTANCES:
            raise DomainError("Choose a scalar series with 1–3,000 images.")
        contents: list[bytes] = []
        total = 0
        for index, uid in enumerate(identifiers):
            progress(index / len(identifiers))
            content, content_type = self.get(
                path + "/" + uid,
                accept=(
                    'multipart/related; type="application/dicom"; '
                    "transfer-syntax=1.2.840.10008.1.2.1"
                ),
                limit=MAX_BYTES - total,
            )
            if content_type.lower().startswith("multipart/"):
                message = BytesParser(policy=policy.default).parsebytes(
                    f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + content
                )
                parts = list(message.iter_parts())
                if len(parts) != 1 or parts[0].get_content_type() != "application/dicom":
                    raise DomainError("Expected one DICOM instance in the server response.")
                payload = parts[0].get_payload(decode=True)
                if not isinstance(payload, bytes):
                    raise DomainError("Could not decode the DICOM instance response.")
                content = payload
            elif not content_type.lower().startswith("application/dicom"):
                raise DomainError("The server did not return a DICOM instance.")
            try:
                header = pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True)
                identity = (
                    str(header.StudyInstanceUID),
                    str(header.SeriesInstanceUID),
                    str(header.SOPInstanceUID),
                )
            except (AttributeError, ValueError, pydicom.errors.InvalidDicomError) as exc:
                raise DomainError("The server returned an invalid DICOM instance.") from exc
            if identity != (ref.study_uid, ref.series_uid, uid):
                raise DomainError(
                    "The downloaded instance does not match the requested DICOM UIDs."
                )
            total += len(content)
            contents.append(content)
        return contents
