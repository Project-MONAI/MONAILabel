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
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException

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
    """
    Retrieve label metadata for an image and tag.

    Parameters:
        image_id (str): Identifier of the image.
        tag (str): Label tag used to locate the metadata.

    Returns:
        Dict[str, Any]: The label metadata when it is a dictionary; otherwise, an empty dictionary.
    """
    try:
        info = datastore.get_label_info(image_id, tag)
        return info if isinstance(info, dict) else {}
    except LabelNotFoundException:
        return {}
    except Exception as exc:
        logger.warning(f"Failed to read label info for {image_id} ({tag}): {exc}")
        return {}


def _parse_date_range(date_range: Optional[str]) -> Optional[Tuple[datetime, datetime]]:
    """
    Parse an optional comma-separated ISO timestamp range.

    Parameters:
        date_range (Optional[str]): Range in the form ``start,end``.

    Returns:
        Optional[Tuple[datetime, datetime]]: The parsed start and end timestamps, or ``None`` when no range is provided.

    Raises:
        HTTPException: If the range is incomplete or contains invalid ISO timestamps.
    """
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
    """
    Determine whether a timestamp falls within an optional inclusive date range.

    Parameters:
        value (Optional[str]): ISO-formatted timestamp to evaluate.
        parsed (Optional[Tuple[datetime, datetime]]): Inclusive start and end timestamps, or `None` to disable filtering.

    Returns:
        bool: `true` if filtering is disabled or the timestamp falls within the range, `false` otherwise.
    """
    if not parsed:
        return True
    if not value:
        return False

    try:
        current = datetime.fromisoformat(value)
    except ValueError:
        return False

    start, end = parsed
    return start <= current <= end


def _review_case(datastore, image_id: str, tag: str) -> Dict[str, Any]:
    """
    Build a review record containing image metadata, review details, and label status.

    Parameters:
        datastore: Datastore used to retrieve image, label, and review information.
        image_id (str): Identifier of the image.
        tag (str): Label tag associated with the review.

    Returns:
        Dict[str, Any]: Review record with derived status, reviewer information, and label presence.
    """
    image_info = datastore.get_image_info(image_id) or {}
    label_info = _safe_label_info(datastore, image_id, tag)
    labels = datastore.get_labels_by_image_id(image_id) or {}

    reviewer = label_info.get("reviewer_name") or label_info.get("reviewer")
    status = label_info.get("status") or ("pending" if labels.get(tag) else "unlabeled")

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
    """
    Collect review cases that match the specified status, image, reviewer, date, and label tag filters.

    Parameters:
        status_filter (Optional[str]): Review status to match.
        search (Optional[str]): Comma-separated image IDs to include.
        reviewer (Optional[str]): Reviewer name to match.
        date_range (Optional[str]): ISO timestamp range in the format ``start,end``.
        tag (str): Label tag used to build each review case.

    Returns:
        List[Dict[str, Any]]: Review cases that satisfy all provided filters.

    Raises:
        HTTPException: If ``date_range`` is not a valid ISO timestamp range.
    """
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
    """
    Count review cases by overall total and status.

    Parameters:
        items (List[Dict[str, Any]]): Review case records containing a status field.

    Returns:
        Dict[str, int]: Counts for total, approved, flagged, pending, and unlabeled cases.
    """
    return {
        "total": len(items),
        "approved": sum(1 for item in items if item.get("status") == "approved"),
        "flagged": sum(1 for item in items if item.get("status") == "flagged"),
        "pending": sum(1 for item in items if item.get("status") in ("pending", "unapproved")),
        "unlabeled": sum(1 for item in items if item.get("status") == "unlabeled"),
    }


def _report_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize review cases by status and difficulty level.

    Parameters:
        items (List[Dict[str, Any]]): Review case records to summarize.

    Returns:
        Dict[str, Any]: Status and difficulty counts with the current recording timestamp.
    """
    return {
        **_summary(items),
        "easy": sum(1 for item in items if item.get("level") == "easy"),
        "medium": sum(1 for item in items if item.get("level") == "medium"),
        "hard": sum(1 for item in items if item.get("level") == "hard"),
        "date_recorded": datetime.now().isoformat(),
    }


def _render_csv(items: List[Dict[str, Any]]) -> str:
    """
    Render review case records as CSV text.

    Parameters:
        items (List[Dict[str, Any]]): Review case records to include in the CSV output.

    Returns:
        str: CSV text containing a header row and one row for each review case.
    """
    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(["image_id", "status", "level", "reviewer", "comment", "last_reviewed", "tag"])
    for item in items:
        writer.writerow(
            [
                item.get("id", ""),
                item.get("status", ""),
                item.get("level", ""),
                item.get("reviewer", ""),
                item.get("comment", ""),
                item.get("last_reviewed", ""),
                item.get("tag", ""),
            ]
        )
    return handle.getvalue()


def _render_html(stats: Dict[str, Any]) -> str:
    """
    Render review statistics as an HTML report.

    Parameters:
        stats (Dict[str, Any]): Statistics and generation timestamp to include in the report.

    Returns:
        str: An HTML document containing review status and difficulty counts with percentages.
    """
    total = stats["total"] or 1

    def row(name: str, value: int) -> str:
        """Render an HTML table row containing a label, count, and percentage of the total.

        Parameters:
                name (str): The label displayed in the row.
                value (int): The count used to calculate the percentage.

        Returns:
                str: An escaped HTML table row with the label, count, and percentage.
        """
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
    """
    List filtered review cases with pagination and aggregate status counts.

    Parameters:
        offset (int): Number of matching cases to skip.
        limit (int): Maximum number of cases to include in the results.
        status_filter (Optional[str]): Status used to filter cases.
        search (Optional[str]): Comma-separated image identifiers used to filter cases.
        reviewer (Optional[str]): Reviewer name used to filter cases.
        date_range (Optional[str]): ISO timestamp range in the format ``start,end``.
        tag (str): Label tag used to build the review cases.

    Returns:
        Dict[str, Any]: A response containing summary counts, paginated results, and filter metadata.
    """
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
    """
    Retrieve the available label versions for an image.

    Parameters:
        image (str): Identifier of the image whose label versions are requested.

    Returns:
        dict: A success response containing the image identifier and version metadata.

    Raises:
        HTTPException: If no labels are found for the image.
    """
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
    """
    Generate a review report in JSON, CSV, or HTML format.

    Parameters:
        fmt (str): Output format: `"json"`, `"csv"`, or `"html"`.
        reviewer (Optional[str]): Limits results to a specific reviewer.
        date_range (Optional[str]): ISO timestamp range in `start,end` format.
        tag (str): Label tag used to select review cases.

    Returns:
        dict: A formatted report containing review statistics and, for JSON output,
            the applied filters and matching review cases.

    Raises:
        HTTPException: If `fmt` is not `"json"`, `"csv"`, or `"html"`.
    """
    items = _list_review_cases(None, None, reviewer, date_range, tag)
    stats = _report_stats(items)
    fmt = fmt.lower()

    if fmt == "csv":
        return {"status": "success", "fmt": "csv", "content": _render_csv(items), "stats": stats}
    if fmt == "html":
        return {"status": "success", "fmt": "html", "content": _render_html(stats), "stats": stats}
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
