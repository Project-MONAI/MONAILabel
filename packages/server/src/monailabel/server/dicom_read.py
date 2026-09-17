"""Read-only DICOMweb projection of authorized, imported workspace series."""

import json
import re
from typing import Any
from uuid import uuid4

from fastapi.responses import Response

from monailabel.core.errors import DomainError
from monailabel.core.models import DicomSeries
from monailabel.dicom.publishing import frame, metadata
from monailabel.dicom.web import value
from monailabel.server.storage import Artifacts

UID = r"[0-9]+(?:\.[0-9]+)*"
RESOURCE = re.compile(
    rf"(?:series|studies(?:/{UID}(?:/series(?:/{UID}(?:/instances(?:/{UID}"
    rf"(?:/frames/1|/metadata)?)?|/metadata)?)?|/metadata)?)?)/?"
)


def dicom_response(
    path: str,
    query: dict[str, str],
    allowed: list[DicomSeries],
    artifacts: Artifacts,
) -> Response:
    if not RESOURCE.fullmatch(path):
        raise DomainError("Unsupported DICOMweb resource.", status=404)
    parts = path.strip("/").split("/")
    study = parts[1] if parts[0] == "studies" and len(parts) > 1 else None
    series = parts[3] if len(parts) > 3 and parts[2] == "series" else None
    instance = parts[5] if len(parts) > 5 and parts[4] == "instances" else None
    sources = [
        s
        for s in allowed
        if (not study or s.study_uid == study) and (not series or s.series_uid == series)
    ]
    if (study or series) and not sources:
        raise DomainError("No access to this imported DICOM study or series.", status=403)
    sources = list({(s.study_uid, s.series_uid): s for s in sources}.values())
    if instance:
        source = next((s for s in sources if instance in s.instance_uids), None)
        if source is None:
            raise DomainError("No access to this imported DICOM instance.", status=403)
        key = source.source_keys[source.instance_uids.index(instance)]
        if parts[-2:] == ["frames", "1"]:
            boundary = "monailabel-" + uuid4().hex
            content = (
                f"--{boundary}\r\nContent-Type: application/octet-stream; "
                "transfer-syntax=1.2.840.10008.1.2.1\r\n\r\n"
            ).encode()
            content += frame(artifacts.path(key)) + f"\r\n--{boundary}--\r\n".encode()
            return Response(
                content,
                media_type=(
                    f'multipart/related; type="application/octet-stream"; boundary={boundary}'
                ),
            )
        if parts[-1] == "metadata":
            return Response(
                json.dumps([metadata(artifacts.path(key))]), media_type="application/dicom+json"
            )
        return Response(artifacts.path(key).read_bytes(), media_type="application/dicom")
    rows: list[dict[str, Any]] = []
    if parts[-1] in {"metadata", "instances"}:
        rows = [metadata(artifacts.path(key)) for source in sources for key in source.source_keys]
    else:
        for source in sources:
            row = metadata(artifacts.path(source.source_keys[0]))
            row["00201209"] = {"vr": "IS", "Value": [len(source.instance_uids)]}
            rows.append(row)
        if path.strip("/") == "studies":
            studies: dict[str, dict[str, Any]] = {}
            for row in rows:
                uid = value(row, "0020000D")
                if uid not in studies:
                    studies[uid] = row
                    row["00080061"] = {"vr": "CS", "Value": []}
                    row["00201206"] = {"vr": "IS", "Value": [0]}
                    row["00201208"] = {"vr": "IS", "Value": [0]}
                study_row = studies[uid]
                modality = value(row, "00080060")
                if modality not in study_row["00080061"]["Value"]:
                    study_row["00080061"]["Value"].append(modality)
                study_row["00201206"]["Value"][0] += 1
                study_row["00201208"]["Value"][0] += row["00201209"]["Value"][0]
            rows = list(studies.values())
    for name, tag in (
        ("StudyInstanceUID", "0020000D"),
        ("SeriesInstanceUID", "0020000E"),
        ("SOPInstanceUID", "00080018"),
    ):
        if requested := query.get(name):
            rows = [row for row in rows if value(row, tag) in requested.split(",")]
    try:
        offset = max(0, int(query.get("offset", "0")))
        limit = max(0, int(query.get("limit", str(len(rows)))))
    except ValueError as exc:
        raise DomainError("DICOMweb limit and offset must be numbers.") from exc
    return Response(json.dumps(rows[offset : offset + limit]), media_type="application/dicom+json")
