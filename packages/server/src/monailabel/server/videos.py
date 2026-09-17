"""Video ingestion and immutable track revisions; image learning stays separate."""

import json
import subprocess
from pathlib import Path

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import Asset, DecisionRequest, Project, ReviewDecision, User
from monailabel.core.video import (
    TrackAnnotation,
    TrackDocument,
    TrackSubmission,
    VideoAsset,
    VideoImport,
    VideoMetadata,
)
from monailabel.server.storage import Artifacts, Store

MAX_VIDEO_BYTES = 2 * 1024**3


def probe_video(path: Path) -> VideoMetadata:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_streams",
                "-show_entries",
                "stream=width,height,codec_name,duration:stream_side_data=rotation:"
                "frame=best_effort_timestamp_time,duration_time,pkt_duration_time",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=180,
            check=True,
        )
        raw = json.loads(result.stdout)
        stream = raw["streams"][0]
        rotation = next(
            (item["rotation"] for item in stream.get("side_data_list", []) if "rotation" in item),
            0,
        )
        if rotation % 360:
            raise DomainError("Rotate this video to upright pixels before importing it.")
        frames = raw["frames"]
        times = [float(frame["best_effort_timestamp_time"]) for frame in frames]
        start = times[0]
        times = [value - start for value in times]
        last = frames[-1]
        step = float(last.get("duration_time", last.get("pkt_duration_time", 0)))
        if step <= 0:
            step = times[-1] - times[-2] if len(times) > 1 else 1 / 30
        return VideoMetadata(
            width=stream["width"],
            height=stream["height"],
            timestamps=times,
            duration=times[-1] + step,
            codec=stream["codec_name"],
        )
    except FileNotFoundError as exc:
        raise DomainError("Install ffmpeg (including ffprobe) to import videos.") from exc
    except (subprocess.SubprocessError, KeyError, IndexError, ValueError) as exc:
        raise DomainError(
            "Could not read video frames and timestamps. Import a valid video clip."
        ) from exc


class Videos:
    def __init__(self, store: Store, artifacts: Artifacts):
        self.store, self.artifacts = store, artifacts

    def import_file(self, project_id: str, request: VideoImport, path: Path) -> VideoAsset:
        self.store.get(Project, project_id)
        if not 0 < path.stat().st_size <= MAX_VIDEO_BYTES:
            raise DomainError("Import a nonempty video no larger than 2 GiB.", status=413)
        metadata = probe_video(path)
        source_key = self.artifacts.put_file(path)
        asset = VideoAsset(
            project_id=project_id,
            name=request.name,
            group_id=request.group_id.strip(),
            split=request.split,
            source_key=source_key,
            metadata_key=self.artifacts.put(metadata.model_dump_json().encode(), "json"),
            width=metadata.width,
            height=metadata.height,
            frames=len(metadata.timestamps),
            duration=metadata.duration,
        )
        if not asset.group_id:
            raise DomainError("Enter a patient or procedure ID for related clips.")
        with self.store.transaction() as session:
            # Frame exports may also be imported as image assets in this project.
            images = session.list(Asset, project_id)
            videos = session.list(VideoAsset, project_id)
            existing: list[Asset | VideoAsset] = [*images, *videos]
            for old in existing:
                if old.group_id == asset.group_id and old.split != asset.split:
                    raise Conflict("All clips and images from a procedure must share one split.")
            for old in videos:
                if old.source_key == source_key:
                    if old.group_id != asset.group_id or old.split != asset.split:
                        raise Conflict(
                            "This video already belongs to a different source group or split."
                        )
                    return old
            session.insert(asset)
        return asset

    def document(self, asset: VideoAsset) -> TrackDocument:
        if not asset.annotation_id:
            return TrackDocument()
        annotation = self.store.get(TrackAnnotation, asset.annotation_id)
        return TrackDocument.model_validate_json(self.artifacts.read(annotation.tracks_key))

    @staticmethod
    def validate(document: TrackDocument, asset: VideoAsset, project: Project) -> None:
        labels = {label.id for label in project.labels if label.id}
        for track in document.tracks:
            if track.label_id not in labels:
                raise DomainError("A track uses a label outside this project.")
            for key in track.keyframes:
                if (
                    key.frame >= asset.frames
                    or key.box[2] > asset.width
                    or key.box[3] > asset.height
                ):
                    raise DomainError("Track geometry or frame is outside this video.")

    def submit(
        self,
        asset_id: str,
        request: TrackSubmission,
        user: User,
        decision: DecisionRequest | None = None,
    ) -> TrackAnnotation:
        with self.store.transaction() as session:
            asset = session.get(VideoAsset, asset_id)
            if asset.revision != request.base_revision:
                raise Conflict(
                    "This video has a newer submitted revision. Your viewer draft is preserved."
                )
            self.validate(request.document, asset, session.get(Project, asset.project_id))
            annotation = TrackAnnotation(
                project_id=asset.project_id,
                asset_id=asset.id,
                revision=asset.revision + 1,
                parent_id=asset.annotation_id,
                created_by=user.id,
                tracks_key=self.artifacts.put(request.document.model_dump_json().encode(), "json"),
            )
            session.insert(annotation)
            session.update(
                asset.model_copy(
                    update={
                        "revision": annotation.revision,
                        "annotation_id": annotation.id,
                    }
                )
            )
            if decision:
                session.insert(
                    ReviewDecision(
                        project_id=asset.project_id,
                        asset_id=asset.id,
                        annotation_id=annotation.id,
                        revision=annotation.revision,
                        reviewer_id=user.id,
                        **decision.model_dump(),
                    )
                )
        return annotation

    def decide(
        self, asset_id: str, revision: int, request: DecisionRequest, user: User
    ) -> ReviewDecision:
        with self.store.transaction() as session:
            asset = session.get(VideoAsset, asset_id)
            if asset.revision != revision or not asset.annotation_id:
                raise Conflict("Review the current submitted video revision.")
            decision = ReviewDecision(
                project_id=asset.project_id,
                asset_id=asset.id,
                annotation_id=asset.annotation_id,
                revision=revision,
                reviewer_id=user.id,
                **request.model_dump(),
            )
            session.insert(decision)
        return decision
