"""CVAT's rectangle/polygon boundary. No workspace persistence or server imports."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from monailabel.core.errors import DomainError
from monailabel.core.models import Label
from monailabel.core.video import (
    ObjectTrack,
    PolygonKeyframe,
    TrackDocument,
    TrackKeyframe,
    VideoAsset,
    VideoKeyframe,
)


def validate_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise DomainError("CVAT URLs must be HTTP(S) origins without credentials or paths.")
    return value.rstrip("/")


def encode_tracks(document: TrackDocument, labels: dict[int, int]) -> dict[str, Any]:
    return {
        "tags": [],
        "shapes": [],
        "tracks": [
            {
                "label_id": labels[track.label_id],
                "frame": track.keyframes[0].frame,
                "attributes": [],
                "shapes": [
                    {
                        "type": "polygon" if isinstance(key, PolygonKeyframe) else "rectangle",
                        "frame": key.frame,
                        "points": key.points if isinstance(key, PolygonKeyframe) else key.box,
                        "outside": key.outside,
                        "occluded": key.occluded,
                        "rotation": 0,
                        "attributes": [],
                    }
                    for key in track.keyframes
                ],
            }
            for track in document.tracks
        ],
    }


def decode_tracks(
    payload: dict[str, Any],
    labels: dict[int, int],
    identities: dict[int, str],
    namespace: str,
) -> TrackDocument:
    """Reject unsupported edits instead of silently discarding annotation data."""
    try:
        if any(payload.get(kind) for kind in ("shapes", "tags", "intervals")):
            raise ValueError("Use rectangle or polygon tracks, not individual shapes or tags.")
        tracks = []
        for track in payload["tracks"]:
            if track.get("elements") or track.get("attributes") or track.get("group"):
                raise ValueError("Track attributes, groups and skeletons are not supported.")
            keys: list[VideoKeyframe] = []
            for shape in track["shapes"]:
                if (
                    shape["type"] not in {"rectangle", "polygon"}
                    or shape.get("rotation", 0) != 0
                    or shape.get("attributes")
                    or shape.get("z_order", 0) != 0
                ):
                    raise ValueError("Use unrotated tracks without custom attributes.")
                key_type = PolygonKeyframe if shape["type"] == "polygon" else TrackKeyframe
                keys.append(
                    key_type(
                        frame=shape["frame"],
                        **{"points" if shape["type"] == "polygon" else "box": shape["points"]},
                        outside=shape["outside"],
                        occluded=shape["occluded"],
                    )
                )
            tracks.append(
                ObjectTrack(
                    id=identities.get(track["id"], f"{namespace}-{track['id']}"),
                    label_id=labels[track["label_id"]],
                    keyframes=keys,
                )
            )
        return TrackDocument(tracks=tracks)
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise DomainError(
            "CVAT annotations contain unsupported tracks, labels or geometry. "
            "Use unrotated rectangle or polygon tracks with no groups or custom attributes. "
            "Your CVAT draft has been kept."
        ) from exc


class CvatClient:
    def __init__(self, url: str, token: str):
        self.url = validate_url(url)
        self.http = httpx.Client(
            base_url=self.url,
            headers={"Authorization": f"Token {token}"},
            timeout=httpx.Timeout(300, connect=10),
            follow_redirects=False,
        )

    def close(self) -> None:
        self.http.close()

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.http.request(method, path, **kwargs)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected an object")
            return result
        except (httpx.HTTPError, ValueError) as exc:
            # Remote response bodies/URLs can include credentials or private file paths.
            raise DomainError(
                "CVAT request failed. Check the configured service, token and task access. "
                "Existing CVAT annotations have not been replaced.",
                status=502,
            ) from exc

    def wait_ready(self) -> None:
        # The public version endpoint can respond before OPA has loaded its rules.
        # Probe an authenticated read; never retry a potentially committed task write.
        deadline = time.monotonic() + 60
        while True:
            try:
                response = self.http.get("/api/tasks", params={"page_size": 1}, timeout=5)
                if response.status_code < 500:
                    response.raise_for_status()
                    return
            except httpx.HTTPStatusError as exc:
                raise DomainError("CVAT task access was denied. Check its service token.") from exc
            except httpx.TransportError:
                pass
            if time.monotonic() >= deadline:
                raise DomainError(
                    "CVAT is still starting. Retry opening the viewer shortly.", status=503
                )
            time.sleep(1)

    def create_task(self, name: str, labels: list[Label]) -> int:
        self.wait_ready()
        result = self.request(
            "POST",
            "/api/tasks",
            json={
                "name": name[:256],
                "segment_size": 0,
                "overlap": 0,
                "labels": [
                    {"name": label.name, "color": label.color, "type": "any"}
                    for label in labels
                    if label.id
                ],
            },
        )
        return int(result["id"])

    def upload(self, task_id: int, path: Path, name: str) -> str:
        with path.open("rb") as stream:
            result = self.request(
                "POST",
                f"/api/tasks/{task_id}/data/",
                data={"image_quality": "100", "use_zip_chunks": "true"},
                files={"client_files[0]": (Path(name).name, stream, "application/octet-stream")},
            )
        return str(result["rq_id"])

    def wait(self, request_id: str, pause: Callable[[], None]) -> None:
        for _ in range(600):
            response = self.request("GET", f"/api/requests/{request_id}")
            if response["status"] == "finished":
                return
            if response["status"] in {"failed", "canceled"}:
                raise DomainError("CVAT could not decode this clip. Check the CVAT import logs.")
            pause()
        raise DomainError("CVAT is still preparing this clip. Retry opening it later.")

    def label_map(self, task_id: int, labels: list[Label]) -> dict[int, int]:
        response = self.request("GET", "/api/labels", params={"task_id": task_id, "page_size": 100})
        by_name = {item["name"]: int(item["id"]) for item in response["results"]}
        try:
            return {label.id: by_name[label.name] for label in labels if label.id}
        except KeyError as exc:
            raise DomainError("CVAT task labels do not match this project.") from exc

    def enable_polygon_labels(self, task_id: int) -> None:
        labels = self.request("GET", "/api/labels", params={"task_id": task_id, "page_size": 100})
        for label in labels["results"]:
            if label.get("type") == "rectangle":
                self.request("PATCH", f"/api/labels/{label['id']}", json={"type": "any"})

    def check_frames(self, task_id: int, asset: VideoAsset) -> None:
        meta = self.request("GET", f"/api/tasks/{task_id}/data/meta")
        if (
            meta["size"] != asset.frames
            or meta["start_frame"] != 0
            or meta["stop_frame"] != asset.frames - 1
            or meta.get("frame_filter")
            or meta.get("deleted_frames")
            or meta.get("included_frames") is not None
            or not meta["frames"]
            or any(
                frame["width"] != asset.width or frame["height"] != asset.height
                for frame in meta["frames"]
            )
        ):
            raise DomainError("CVAT changed the source frame grid. This task cannot be submitted.")

    def job_id(self, task_id: int) -> int:
        result = self.request("GET", "/api/jobs", params={"task_id": task_id})
        jobs = [j for j in result["results"] if j["type"] == "annotation"]
        if len(jobs) != 1 or result.get("next"):
            raise DomainError("The video editor requires one complete annotation job per clip.")
        return int(jobs[0]["id"])

    def annotations(self, task_id: int) -> dict[str, Any]:
        return self.request("GET", f"/api/tasks/{task_id}/annotations")

    def seed(self, task_id: int, document: TrackDocument, labels: dict[int, int]) -> dict[int, str]:
        payload = self.request(
            "PUT", f"/api/tasks/{task_id}/annotations", json=encode_tracks(document, labels)
        )
        # CVAT's serializer preserves the submitted track order.
        if len(payload["tracks"]) != len(document.tracks):
            raise DomainError("CVAT did not preserve the submitted tracks.")
        return {
            remote["id"]: local.id
            for remote, local in zip(payload["tracks"], document.tracks, strict=True)
        }
