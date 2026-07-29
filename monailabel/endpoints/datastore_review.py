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

"""
Reviewer-focused aggregate endpoints built on top of the standard datastore API.
"""

import csv
import io
import logging
from datetime import datetime, timezone
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

from monailabel.config import RBAC_USER, settings
from monailabel.endpoints.user.auth import RBAC, User
from monailabel.interfaces.datastore import DefaultLabelTag
from monailabel.interfaces.exception import LabelNotFoundException
from monailabel.interfaces.utils.app import app_instance

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/review",
    tags=["Review"],
    responses={404: {"description": "Not found"}},
)


def _safe_label_info(datastore, image_id: str, tag: str) -> Dict[str, Any]:
    try:
        info = datastore.get_label_info(image_id, tag)
        return info if isinstance(info, dict) else {}
    except LabelNotFoundException:
        return {}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_date_range(date_range: Optional[str]) -> Optional[Tuple[datetime, datetime]]:
    if not date_range:
        return None

    parts = [part.strip() for part in date_range.split(",", maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise HTTPException(status_code=400, detail="date_range must be 'start,end' in ISO format")

    try:
        return datetime.fromisoformat(parts[0]), datetime.fromisoformat(parts[1])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date_range must use ISO timestamps") from exc


def _matches_date_range(value: Optional[str], parsed: Optional[Tuple[datetime, datetime]]) -> bool:
    if not parsed:
        return True
    if not value:
        return False

    try:
        current = datetime.fromisoformat(value)
    except ValueError:
        return False

    start, end = parsed
    current = _as_utc(current)
    start = _as_utc(start)
    end = _as_utc(end)
    return start <= current <= end


def _review_case(datastore, image_id: str, tag: str) -> Dict[str, Any]:
    image_info = datastore.get_image_info(image_id) or {}
    labels = datastore.get_labels_by_image_id(image_id) or {}
    label_id = labels.get(tag) or image_id
    label_info = _safe_label_info(datastore, label_id, tag)

    reviewer = label_info.get("reviewer_name") or label_info.get("reviewer")
    raw_status = label_info.get("status")
    status = raw_status.lower() if isinstance(raw_status, str) else ("pending" if labels.get(tag) else "unlabeled")

    return {
        "id": image_id,
        "name": image_info.get("name", image_id),
        "path": image_info.get("path"),
        "status": status,
        "level": label_info.get("level"),
        "comment": label_info.get("comment"),
        "review_count": label_info.get("review_count", 0),
        "last_reviewed": label_info.get("last_reviewed"),
        "reviewer": reviewer,
        "reviewer_name": reviewer,
        "tag": tag,
        "has_label": bool(labels.get(tag)),
    }


def _list_review_cases(
    status_filter: Optional[str],
    search: Optional[str],
    reviewer: Optional[str],
    date_range: Optional[str],
    tag: str,
) -> List[Dict[str, Any]]:
    datastore = app_instance().datastore()
    image_ids = datastore.list_images()
    parsed_range = _parse_date_range(date_range)

    selected_ids = None
    if search:
        selected_ids = {item.strip() for item in search.split(",") if item.strip()}

    results: List[Dict[str, Any]] = []
    for image_id in image_ids:
        if selected_ids and image_id not in selected_ids:
            continue

        item = _review_case(datastore, image_id, tag)
        if status_filter and item["status"] != status_filter.lower():
            continue
        if reviewer and item.get("reviewer") != reviewer:
            continue
        if not _matches_date_range(item.get("last_reviewed"), parsed_range):
            continue
        results.append(item)

    return results


def _summary(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(items),
        "approved": sum(1 for item in items if item.get("status") == "approved"),
        "flagged": sum(1 for item in items if item.get("status") == "flagged"),
        "pending": sum(1 for item in items if item.get("status") in ("pending", "unapproved")),
        "unlabeled": sum(1 for item in items if item.get("status") == "unlabeled"),
    }


def _report_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        **_summary(items),
        "easy": sum(1 for item in items if item.get("level") == "easy"),
        "medium": sum(1 for item in items if item.get("level") == "medium"),
        "hard": sum(1 for item in items if item.get("level") == "hard"),
        "date_recorded": datetime.now().isoformat(),
    }


def _render_csv(items: List[Dict[str, Any]]) -> str:
    def sanitize_csv_value(value: Any) -> str:
        """Sanitize CSV values to prevent formula injection."""
        text = str(value) if value else ""
        # Prefix values that start with formula-like characters
        if text and text[0] in ("=", "+", "-", "@"):
            return "'" + text
        return text

    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(["image_id", "status", "level", "reviewer", "comment", "last_reviewed", "tag"])
    for item in items:
        writer.writerow(
            [
                sanitize_csv_value(item.get("id", "")),
                sanitize_csv_value(item.get("status", "")),
                sanitize_csv_value(item.get("level", "")),
                sanitize_csv_value(item.get("reviewer", "")),
                sanitize_csv_value(item.get("comment", "")),
                sanitize_csv_value(item.get("last_reviewed", "")),
                sanitize_csv_value(item.get("tag", "")),
            ]
        )
    return handle.getvalue()


def _render_html(stats: Dict[str, Any]) -> str:
    total = stats["total"] or 1

    def row(name: str, value: int) -> str:
        return f"<tr><td>{escape(name)}</td><td>{value}</td>" f"<td>{(100.0 * value / total):.1f}%</td></tr>"

    return (
        "<html><head><title>MONAILabel Review Report</title></head><body>"
        f"<h1>Review Statistics (Total: {stats['total']})</h1>"
        "<table border='1'>"
        "<tr><th>Status</th><th>Count</th><th>Percentage</th></tr>"
        f"{row('Approved', stats['approved'])}"
        f"{row('Flagged', stats['flagged'])}"
        f"{row('Pending', stats['pending'])}"
        f"{row('Unlabeled', stats['unlabeled'])}"
        f"{row('Easy', stats['easy'])}"
        f"{row('Medium', stats['medium'])}"
        f"{row('Hard', stats['hard'])}"
        "</table>"
        f"<p>Generated: {escape(stats['date_recorded'])}</p>"
        "</body></html>"
    )


@router.get("/cases", summary=f"{RBAC_USER}List review cases")
async def api_review_cases(
    offset: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    search: Optional[str] = None,
    reviewer: Optional[str] = None,
    date_range: Optional[str] = None,
    tag: str = DefaultLabelTag.FINAL.value,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER)),
):
    items = _list_review_cases(status_filter, search, reviewer, date_range, tag)
    results = items[offset : offset + limit]

    return {
        "summary": _summary(items),
        "results": results,
        "metadata": {
            "tag": tag,
            "offset": offset,
            "limit": limit,
            "reviewer": reviewer,
            "date_range": date_range,
        },
    }


@router.get("/versions", summary=f"{RBAC_USER}List label versions for an image")
async def api_review_versions(
    image: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER)),
):
    datastore = app_instance().datastore()
    labels = datastore.get_labels_by_image_id(image)
    if not labels:
        raise HTTPException(status_code=404, detail=f"No labels found for image '{image}'")

    versions = []
    for tag, label_id in labels.items():
        info = _safe_label_info(datastore, label_id, tag)
        versions.append(
            {
                "tag": tag,
                "label": label_id,
                "author": info.get("reviewer_name") or info.get("reviewer") or info.get("model"),
                "created_at": info.get("created_at") or info.get("ts"),
                "review_status": info.get("status", "pending"),
                "review_count": info.get("review_count", 0),
            }
        )

    return {"status": "success", "image_id": image, "versions": versions}


@router.get("/report", summary=f"{RBAC_USER}Generate review report")
async def api_review_report(
    fmt: str = "json",
    reviewer: Optional[str] = None,
    date_range: Optional[str] = None,
    tag: str = DefaultLabelTag.FINAL.value,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER)),
):
    items = _list_review_cases(None, None, reviewer, date_range, tag)
    stats = _report_stats(items)
    fmt = fmt.lower()

    if fmt == "csv":
        return PlainTextResponse(_render_csv(items), media_type="text/csv")
    if fmt == "html":
        return HTMLResponse(_render_html(stats))
    if fmt != "json":
        raise HTTPException(status_code=400, detail="fmt must be one of: json, csv, html")

    return {
        "status": "success",
        "fmt": "json",
        "report": stats,
        "date_generated": datetime.now().isoformat(),
        "filters": {"reviewer": reviewer, "date_range": date_range, "tag": tag},
        "reviews": items,
    }
