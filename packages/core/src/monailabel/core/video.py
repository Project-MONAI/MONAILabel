"""Video and temporal annotation contracts, independent of a viewer or tracker."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from monailabel.core.models import Contract, Record, Split, new_id

Coordinate = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class VideoMetadata(Contract):
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    # Presentation-order timestamps from the decoder, not index / nominal FPS.
    timestamps: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        min_length=1, max_length=200_000
    )
    duration: float = Field(gt=0, allow_inf_nan=False)
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


class ObjectTrack(Contract):
    id: str = Field(default_factory=new_id, pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    label_id: int = Field(gt=0, le=255)
    keyframes: list[TrackKeyframe] = Field(min_length=1, max_length=200_000)

    @model_validator(mode="after")
    def ordered_frames(self) -> Self:
        frames = [key.frame for key in self.keyframes]
        if frames != sorted(set(frames)):
            raise ValueError("A track requires unique keyframes in presentation order.")
        if all(key.outside for key in self.keyframes):
            raise ValueError("A track must have at least one visible keyframe.")
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
