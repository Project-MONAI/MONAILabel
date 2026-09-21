"""Revision-bound proposals; inference never writes a CVAT draft."""

import io
import json
import subprocess
import zipfile

import numpy as np

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import Job, ModelRecord, Project
from monailabel.core.ports import ToolDetector, VideoTracker, VideoTrackingResult
from monailabel.core.video import (
    PolygonKeyframe,
    TrackKeyframe,
    VideoAsset,
    VideoDetectionProvenance,
    VideoFindTrackingRequest,
    VideoKeyframe,
    VideoTrackingProposal,
    VideoTrackingRequest,
)
from monailabel.providers.sam import MODELS
from monailabel.providers.tool_detection import RemoteToolDetector
from monailabel.providers.video_geometry import box_from_mask, mask_png, polygon_from_mask
from monailabel.providers.vision import VISION_PROVIDERS
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.models import Models
from monailabel.server.storage import Artifacts, Store
from monailabel.server.video.models import VideoEditor


class VideoTracking:
    def __init__(self, store: Store, artifacts: Artifacts, jobs: Jobs, models: Models):
        self.store, self.artifacts, self.jobs = store, artifacts, jobs
        self.models = models
        self.provider: VideoTracker | None = None
        self.detector: ToolDetector = RemoteToolDetector(models.credentials)

    def supports(self, model: ModelRecord) -> bool:
        return (
            not self.models.requires_3d(model)
            and not self.models.requires_spatial(model)
            and (model.provider in self.models.providers or model.provider == "monai-unet")
        )

    def validate(
        self, video_id: str, request: VideoTrackingRequest | VideoFindTrackingRequest
    ) -> VideoAsset:
        asset = self.store.get(VideoAsset, video_id)
        editor = self.store.get(VideoEditor, request.editor_id)
        if editor.asset_id != asset.id or editor.project_id != asset.project_id or not editor.ready:
            raise DomainError("Choose a ready CVAT editor for this video.")
        if (
            asset.revision != request.base_revision
            or editor.base_revision != asset.revision
            or editor.submitted_annotation_id
        ):
            raise Conflict(
                "This video revision changed. Reload before tracking; your CVAT draft is preserved."
            )
        if request.label_id not in editor.label_map or request.label_id not in {
            label.id for label in self.store.get(Project, asset.project_id).labels
        }:
            raise DomainError("Choose an instrument label in this project.")
        frame = request.seed.frame if isinstance(request, VideoTrackingRequest) else request.frame
        if frame + request.frame_count > asset.frames:
            raise DomainError("Choose a visible tool and a tracking range within the clip.")
        if isinstance(request, VideoTrackingRequest):
            if request.seed.outside:
                raise DomainError("Choose a visible tool to track.")
            if request.seed.box[2] > asset.width or request.seed.box[3] > asset.height:
                raise DomainError("The tool annotation extends beyond the source image.")
        return asset

    def start(self, video_id: str, request: VideoTrackingRequest) -> Job:
        if request.client_id is None:
            raise DomainError("Select a rectangle or polygon track, or ask to locate a tool.")
        asset = self.validate(video_id, request)
        return self.jobs.submit(
            "video_tracking",
            asset.project_id,
            {"asset_id": asset.id, **request.model_dump(mode="json")},
            lambda context: self.run(asset, request, context),
        )

    def find(self, video_id: str, request: VideoFindTrackingRequest) -> Job:
        asset = self.validate(video_id, request)
        model = self.models.get(asset.project_id, request.model_id)
        if not self.supports(model):
            raise DomainError(
                "Choose a configured 2D annotation model to locate or segment the tool."
            )

        def run(context: JobContext) -> Outcome:
            self.validate(video_id, request)
            self.models.get(asset.project_id, model.id)
            context.progress(
                0.01, f"Annotating the tool with {model.name} on source frame {request.frame}."
            )
            try:
                decoded = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-fflags",
                        "+genpts",
                        "-i",
                        str(self.artifacts.path(asset.source_key)),
                        "-map",
                        "0:v:0",
                        "-vf",
                        f"select=eq(n\\,{request.frame})",
                        "-frames:v",
                        "1",
                        "-fps_mode",
                        "passthrough",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "pipe:1",
                    ],
                    capture_output=True,
                    check=True,
                    timeout=120,
                )
                image = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(
                    asset.height, asset.width, 3
                )
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                raise DomainError("Could not decode the selected source video frame.") from exc
            project = self.store.get(Project, asset.project_id)
            label = next(label for label in project.labels if label.id == request.label_id)
            image_float = image.astype(np.float32) / np.float32(255)
            seed: VideoKeyframe | None
            seed_mask = None
            seed_warnings: list[str] = []
            if request.output == "box" and model.provider in VISION_PROVIDERS:
                detection = self.detector.locate(image_float, label, request.prompt, model)
                box = detection.box
                reason = (
                    "could not locate the tool"
                    if detection.status == "not_found"
                    else "found multiple possible tools"
                )
                seed = TrackKeyframe(frame=request.frame, box=box) if box else None
            else:
                if label.id not in self.models.supported_labels(project, model):
                    raise DomainError("The chosen model does not support this tool label.")
                selected_model = self.models.for_labels(model, [label.id])
                mask = self.models.predict(
                    project,
                    selected_model,
                    image_float,
                    "Segment only one visible instance of " + label.name + ". "
                    "Exclude surrounding tissue, image borders and overlays. "
                    "Return no segmentation if absent or ambiguous. " + request.prompt,
                )
                binary = (mask == label.id).astype(np.uint8)
                if not binary.any():
                    seed = None
                elif request.output == "polygon":
                    outline, complex_shape = polygon_from_mask(binary, request.frame)
                    seed = outline
                    seed_mask = mask_png(binary)
                    if complex_shape:
                        seed_warnings.append(
                            "The starting segmentation has holes or disconnected regions; "
                            "its polygon shows the largest outer outline. "
                            "Original masks are retained."
                        )
                else:
                    seed = box_from_mask(binary, request.frame)
                reason = "could not segment a visible tool"
            if seed is None:
                raise DomainError(
                    f"{model.name} {reason}. Choose another frame, describe one tool precisely, "
                    "or draw a starting box or polygon. No track was created."
                )
            tracking = VideoTrackingRequest(
                editor_id=request.editor_id,
                base_revision=request.base_revision,
                client_id=None,
                label_id=request.label_id,
                seed=seed,
                output=request.output,
                frame_count=request.frame_count,
                draft_signature=request.draft_signature,
            )
            self.validate(video_id, tracking)
            provenance = VideoDetectionProvenance(
                model_id=model.id,
                model_name=model.name,
                model_version=model.version,
                provider=model.provider,
                remote_model=str(model.config.get("model", "")),
                prompt=request.prompt,
            )
            return self.run(asset, tracking, context, provenance, seed_mask, seed_warnings)

        return self.jobs.submit(
            "video_find_tracking",
            asset.project_id,
            {"asset_id": asset.id, **request.model_dump(mode="json")},
            run,
        )

    def run(
        self,
        asset: VideoAsset,
        request: VideoTrackingRequest,
        context: JobContext,
        detection: VideoDetectionProvenance | None = None,
        seed_mask: bytes | None = None,
        seed_warnings: list[str] | None = None,
    ) -> Outcome:
        annotation_only = request.frame_count == 1 and (
            (request.output == "polygon") == isinstance(request.seed, PolygonKeyframe)
        )
        if annotation_only:
            context.progress(0.95, "Preparing the single-frame annotation proposal.")
            result = VideoTrackingResult(
                [request.seed], {request.seed.frame: seed_mask} if seed_mask else {}, []
            )
        else:
            context.progress(
                0.01,
                "Tracking the selected tool with SAM 2.1. The first run downloads model weights.",
            )
            if self.provider is None:
                from monailabel.sam.video import SamVideoTracker

                self.provider = SamVideoTracker()
            result = self.provider.track(
                self.artifacts.path(asset.source_key),
                asset.width,
                asset.height,
                request.seed,
                request.frame_count,
                context.progress,
                request.output,
                seed_mask,
            )
        keyframes = result.keyframes
        self.validate(asset.id, request)
        if [k.frame for k in keyframes] != list(
            range(request.seed.frame, request.seed.frame + request.frame_count)
        ) or any(k.box[2] > asset.width or k.box[3] > asset.height for k in keyframes):
            raise DomainError("The tracker returned invalid source-frame geometry.")
        spec = MODELS["sam2"]
        expected_type = PolygonKeyframe if request.output == "polygon" else TrackKeyframe
        if any(not isinstance(key, expected_type) for key in keyframes):
            raise DomainError("The tracker returned the wrong annotation shape.")
        if not isinstance(request.seed, expected_type):
            # Changing shape adds a new track, preserving the selected native object.
            request = request.model_copy(update={"client_id": None})
        masks_key = None
        if result.masks:
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "metadata.json",
                    json.dumps(
                        {
                            "width": asset.width,
                            "height": asset.height,
                            "frames": sorted(result.masks),
                            "source_frame_numbers": True,
                        }
                    ),
                )
                for frame, content in result.masks.items():
                    bundle.writestr(f"{frame:06d}.png", content)
            masks_key = self.artifacts.put(archive.getvalue())
        proposal = VideoTrackingProposal(
            project_id=asset.project_id,
            asset_id=asset.id,
            request=request,
            keyframes=keyframes,
            provider=(detection.provider if detection else "manual") if annotation_only else "sam2",
            model_revision=None if annotation_only else spec.revision,
            model_checksum=None if annotation_only else spec.checksum,
            detection=detection,
            masks_key=masks_key,
            warnings=(seed_warnings or []) + result.warnings,
        )
        return Outcome(result={"video_proposal_id": proposal.id}, records=[proposal])
