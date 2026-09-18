"""Video and temporal annotation contracts, independent of a viewer or tracker."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from monailabel.core.models import Contract, DecisionRequest, Record, Split, new_id

Coordinate = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class VideoMetadata(Contract):
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    # Presentation-order timestamps from the decoder, not index / nominal FPS.
    timestamps: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        min_length=1, max_length=200_000
    )
    duration: float = Field(gt=0, allow_inf_nan=False)
    # Original presentation timestamp of the first frame, before zero normalization.
    start_time: float = Field(default=0, allow_inf_nan=False)
    codec: str
    rotation: Literal[0] = 0

    @model_validator(mode="after")
    def ordered_frames(self) -> Self:
        if self.timestamps[0] != 0 or any(
            a >= b for a, b in zip(self.timestamps, self.timestamps[1:], strict=False)
        ):
            raise ValueError("Frame timestamps must start at zero and strictly increase.")
        if self.duration <= self.timestamps[-1]:
            raise ValueError("Video duration must include the last frame.")
        return self


class VideoAsset(Record):
    project_id: str
    name: str
    group_id: str
    split: Split
    kind: Literal["video"] = "video"
    source_key: str
    metadata_key: str
    width: int
    height: int
    frames: int
    duration: float
    revision: int = 0
    annotation_id: str | None = None


class TrackKeyframe(Contract):
    frame: int = Field(ge=0)
    # Continuous image edge coordinates: left, top, right, bottom.
    box: list[Coordinate] = Field(min_length=4, max_length=4)
    outside: bool = False
    occluded: bool = False

    @model_validator(mode="after")
    def rectangle(self) -> Self:
        if self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
            raise ValueError("Track rectangles must have positive width and height.")
        return self


class PolygonKeyframe(Contract):
    frame: int = Field(ge=0)
    points: list[Coordinate] = Field(min_length=6, max_length=4096)
    outside: bool = False
    occluded: bool = False

    @property
    def box(self) -> list[float]:
        return [
            min(self.points[::2]),
            min(self.points[1::2]),
            max(self.points[::2]),
            max(self.points[1::2]),
        ]

    @model_validator(mode="after")
    def polygon(self) -> Self:
        if len(self.points) % 2:
            raise ValueError("Polygon coordinates must be x/y pairs.")
        vertices = list(zip(self.points[::2], self.points[1::2], strict=True))
        area = sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=True)
        )
        if abs(area) < 1e-6:
            raise ValueError("Polygon must enclose a nonzero area.")
        return self


VideoKeyframe = TrackKeyframe | PolygonKeyframe


class ObjectTrack(Contract):
    id: str = Field(default_factory=new_id, pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    label_id: int = Field(gt=0, le=255)
    keyframes: list[VideoKeyframe] = Field(min_length=1, max_length=200_000)

    @model_validator(mode="after")
    def ordered_frames(self) -> Self:
        frames = [key.frame for key in self.keyframes]
        if frames != sorted(set(frames)):
            raise ValueError("A track requires unique keyframes in presentation order.")
        if all(key.outside for key in self.keyframes):
            raise ValueError("A track must have at least one visible keyframe.")
        if len({type(key) for key in self.keyframes}) != 1:
            raise ValueError("A track cannot mix rectangle and polygon keyframes.")
        return self


class TrackDocument(Contract):
    tracks: list[ObjectTrack] = Field(default_factory=list, max_length=1000)
    interpolation: Literal["linear"] = "linear"

    @model_validator(mode="after")
    def identities(self) -> Self:
        if len({track.id for track in self.tracks}) != len(self.tracks):
            raise ValueError("Each tracked instance requires a unique ID.")
        return self


class TrackAnnotation(Record):
    project_id: str
    asset_id: str
    revision: int
    parent_id: str | None = None
    tracks_key: str
    created_by: str


class TrackSubmission(Contract):
    base_revision: int = Field(ge=0)
    document: TrackDocument


class VideoImport(Contract):
    name: str = Field(min_length=1, max_length=200)
    group_id: str = Field(min_length=1, max_length=200)
    split: Split = Split.POOL
    labels: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        default_factory=list, max_length=31
    )


class VideoRevision(Contract):
    base_revision: int = Field(ge=0)


class VideoDecision(VideoRevision, DecisionRequest):
    pass


class VideoEditorRequest(VideoRevision):
    mode: Literal["annotation", "review"] = "annotation"


class VideoTrackingRequest(VideoRevision):
    editor_id: str
    client_id: int | None = Field(ge=0)
    label_id: int = Field(gt=0, le=255)
    seed: VideoKeyframe
    output: Literal["box", "polygon"] = "box"
    frame_count: int = Field(default=16, ge=1, le=200_000)
    draft_signature: str = Field(pattern=r"^[a-f0-9]{64}$")


class VideoFindTrackingRequest(VideoRevision):
    editor_id: str
    model_id: str
    label_id: int = Field(gt=0, le=255)
    prompt: str = Field(default="", max_length=2000)
    output: Literal["box", "polygon"] = "box"
    frame: int = Field(ge=0)
    frame_count: int = Field(default=16, ge=1, le=200_000)
    draft_signature: str = Field(pattern=r"^[a-f0-9]{64}$")


class ToolDetection(Contract):
    status: Literal["found", "not_found", "ambiguous"]
    box: list[Coordinate] | None = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def located(self) -> Self:
        if (self.status == "found") != (self.box is not None):
            raise ValueError("Only a found tool has a box.")
        if self.box is not None:
            TrackKeyframe(frame=0, box=self.box)
        return self


class VideoDetectionProvenance(Contract):
    model_id: str
    model_name: str
    model_version: int
    provider: str
    remote_model: str
    prompt: str


class VideoTrackingProposal(Record):
    project_id: str
    asset_id: str
    request: VideoTrackingRequest
    keyframes: list[VideoKeyframe]
    provider: str = "sam2"
    model_revision: str | None = None
    model_checksum: str | None = None
    detection: VideoDetectionProvenance | None = None
    masks_key: str | None = None
    warnings: list[str] = Field(default_factory=list)
