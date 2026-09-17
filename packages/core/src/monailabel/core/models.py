"""Versioned data contracts shared by the API, workers, and clients."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from monailabel.core.colors import default_color


def new_id() -> str:
    return uuid4().hex


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Record(Contract):
    id: str = Field(default_factory=new_id)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Label(Contract):
    id: int = Field(ge=0, le=255)
    name: str = Field(min_length=1, max_length=80)
    color: str = Field(
        default_factory=lambda data: default_color(data["name"], data["id"]),
        pattern=r"^#[0-9a-fA-F]{6}$",
        description="Optional display override; omit to use the anatomical palette.",
    )


class ProjectCreate(Contract):
    name: str = Field(min_length=1, max_length=120)
    labels: list[Label] = Field(
        default_factory=lambda: [Label(id=0, name="Background", color="#000000")],
        description="Omit to start with only Background. If supplied, include Background with "
        "id=0 and at least one entry; an empty list is invalid.",
        min_length=1,
        max_length=32,
    )
    instructions: str = Field(default="", max_length=8000)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        ids = [label.id for label in self.labels]
        names = [label.name.casefold() for label in self.labels]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names) or 0 not in ids:
            raise ValueError("Labels require unique IDs/names and label 0 for background.")
        return self


class Project(ProjectCreate, Record):
    defaults: dict[int, str] = Field(default_factory=dict)
    annotation_model_id: str | None = None
    version: int = 0
    protocol_version: int = 1
    is_demo: bool = False


class ProjectUpdate(Contract):
    name: str = Field(min_length=1, max_length=120)
    instructions: str = Field(default="", max_length=8000)
    base_version: int = Field(ge=0)


class LabelColorsUpdate(Contract):
    targets: list[str] = Field(min_length=1, max_length=31)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    base_version: int = Field(ge=0)


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    POOL = "pool"


class DeleteProjectRequest(Contract):
    confirmation_name: str = Field(min_length=1, max_length=200)


class DeleteAssetsRequest(Contract):
    asset_ids: list[str] = Field(min_length=1, max_length=10000)


class DeletionResult(Contract):
    project_id: str
    asset_ids: list[str]
    project_deleted: bool = False
    storage_cleanup: Literal["next_restart"] = "next_restart"


class ImageMetadata(Contract):
    name: str = Field(min_length=1, max_length=200)
    group_id: str | None = Field(default=None, min_length=1, max_length=200)
    split: Split = Split.POOL
    shared: bool = False


class ImageImport(ImageMetadata):
    image_base64: str = Field(max_length=180_000_000)


class Asset(Record):
    project_id: str
    name: str
    group_id: str
    split: Split
    image_key: str
    spatial_shape: list[int]
    kind: Literal["image2d", "volume3d"] = "image2d"
    affine: list[list[float]] | None = None
    source_key: str | None = None
    revision: int = 0
    annotation_id: str | None = None
    fixture_key: str | None = None


class DicomSeries(Record):
    project_id: str
    asset_id: str
    study_uid: str
    series_uid: str
    frame_of_reference_uid: str
    instance_uids: list[str]
    source_keys: list[str]
    orthanc_series_id: str = ""
    connection_id: str | None = None
    derived_from_nifti: bool = False
    source_group_id: str | None = None


class ModelRecord(Record):
    project_id: str | None = None
    name: str
    provider: str
    label_ids: list[int]
    config: dict[str, JsonValue] = Field(default_factory=dict)
    state_key: str | None = None
    learner_id: str | None = None
    parent_id: str | None = None
    snapshot_id: str | None = None
    training_assets: list[str] = Field(default_factory=list)
    training_groups: list[str] = Field(default_factory=list)
    training_revisions: dict[str, int] = Field(default_factory=dict)
    mode: str | None = None
    preset: str | None = None
    read_only: bool = False
    inherit_targets: bool = False
    unreviewed_training: bool = False
    version: int = 0
    archived: bool = False


class ModelRegister(Contract):
    name: str = Field(min_length=1, max_length=120)
    provider: Literal["http-mask", "openai-polygons", "openai-chat-polygons", "huggingface"]
    label_ids: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=32)
    config: dict[str, JsonValue] = Field(default_factory=dict)


class ModelUpdate(Contract):
    name: str = Field(min_length=1, max_length=120)
    base_version: int = Field(ge=0)


class DeleteModelRequest(Contract):
    confirmation_name: str
    base_version: int = Field(ge=0)
    scope: Literal["item", "model"] = "item"
    related_versions: dict[str, int] | None = None


class PlaneOrientation(Contract):
    transpose: bool = False
    flip_rows: bool = False
    flip_columns: bool = False


class SliceScope(Contract):
    axis: int = Field(ge=0, le=2)
    index: int = Field(ge=0)
    window: list[float] | None = Field(default=None, min_length=2, max_length=2)
    orientation: PlaneOrientation = Field(default_factory=PlaneOrientation)


class PromptPoint(Contract):
    coordinates: list[Annotated[float, Field(allow_inf_nan=False, ge=0)]] = Field(
        min_length=2, max_length=3
    )
    positive: bool = True


class SpatialPrompt(Contract):
    """One object's hints in source array coordinates: row/column or I/J/K.

    Box corners are inclusive voxel/pixel centers. These coordinates come from
    the viewer, never from the language model or a reference segmentation.
    """

    points: list[PromptPoint] = Field(default_factory=list, max_length=64)
    box: list[list[Annotated[float, Field(allow_inf_nan=False, ge=0)]]] | None = Field(
        default=None, min_length=2, max_length=2
    )

    @model_validator(mode="after")
    def geometry(self) -> Self:
        coordinates = [point.coordinates for point in self.points] + (self.box or [])
        if not coordinates or len({len(point) for point in coordinates}) != 1:
            raise ValueError("Provide a box or points with matching source dimensions.")
        if len(coordinates[0]) not in {2, 3}:
            raise ValueError("Spatial hints must use 2D pixels or 3D source voxels.")
        if self.box and any(a > b for a, b in zip(*self.box, strict=True)):
            raise ValueError("Box corners must be ordered from low to high.")
        if not self.box and not any(point.positive for point in self.points):
            raise ValueError("Include at least one positive point or a box.")
        return self


class SpatialObject(Contract):
    """Editable viewer hint, in source IJK coordinates, independent of mask labels."""

    id: str = Field(min_length=1, max_length=160)
    target: str = Field(default="", max_length=80)
    kind: Literal["box", "point"]
    coordinates: list[list[Annotated[float, Field(allow_inf_nan=False, ge=0)]]] = Field(
        min_length=1, max_length=2
    )
    positive: bool = True
    selected: bool = False

    @model_validator(mode="after")
    def geometry(self) -> Self:
        if any(len(point) != 3 for point in self.coordinates):
            raise ValueError("Viewer hints require source IJK coordinates.")
        if len(self.coordinates) != (2 if self.kind == "box" else 1):
            raise ValueError("Provide one point or two box corners.")
        if self.kind == "box" and any(a > b for a, b in zip(*self.coordinates, strict=True)):
            raise ValueError("Box corners must be ordered low to high.")
        return self


class SpatialEditAction(Contract):
    client_action: Literal["edit_spatial_prompts"] = "edit_spatial_prompts"
    project_id: str
    asset_id: str
    base_revision: int = Field(ge=0)
    slice: SliceScope
    expected: list[SpatialObject] = Field(max_length=128)
    upsert: list[SpatialObject] = Field(default_factory=list, max_length=128)
    remove: list[str] = Field(default_factory=list, max_length=128)


class BoxRequest(Contract):
    model_id: str
    target: str = Field(min_length=1, max_length=80)
    prompt: str = Field(default="", max_length=8000)
    slice: SliceScope


class RoiRequest(BoxRequest):
    end_index: int = Field(ge=0)  # Inclusive, zero-based source index.


class RegionProposal(Record):
    project_id: str
    asset_id: str
    base_revision: int
    model_id: str
    target: str
    prompt: str
    slice: SliceScope
    end_index: int | None = None  # Set for a 3D ROI; None for a slice-local box.
    detected_slices: int = 0
    bounds: list[list[int]]  # Empty for no target; otherwise two source-IJK corners.


class ImageRegion(Contract):
    """Source-image pixel crop, optionally restricted by row-major [start, stop) runs."""

    x: int = Field(strict=True, ge=0)
    y: int = Field(strict=True, ge=0)
    width: int = Field(strict=True, gt=0)
    height: int = Field(strict=True, gt=0)
    runs: (
        list[tuple[Annotated[int, Field(strict=True)], Annotated[int, Field(strict=True)]]] | None
    ) = Field(default=None, min_length=1, max_length=65536)

    @model_validator(mode="after")
    def validate_runs(self) -> Self:
        area = self.width * self.height
        if area > 16_777_216:
            raise ValueError("Selected region exceeds the 2D pixel limit.")
        previous = 0
        for start, stop in self.runs or []:
            if not previous <= start < stop <= area:
                raise ValueError("Region runs must be ordered, nonoverlapping crop pixel ranges.")
            previous = stop
        return self


class ImageTiling(Contract):
    tile_size: int = Field(default=256, strict=True, ge=64, le=2048)
    overlap: int = Field(default=32, strict=True, ge=0, le=256)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.overlap * 4 > self.tile_size:
            raise ValueError("Tile context on each edge cannot exceed one quarter of tile_size.")
        return self


class ClearSegmentsAction(Contract):
    """A revision-checked local edit; saved annotation revisions remain immutable."""

    client_action: Literal["clear_segments"] = "clear_segments"
    project_id: str
    asset_id: str
    base_revision: int = Field(ge=0)
    label_ids: list[int] = Field(min_length=1)
    image_region: ImageRegion | None = None
    slice: SliceScope | None = None


class ClassificationObject(Contract):
    id: str = Field(min_length=1, max_length=80)
    label_id: int = Field(strict=True, ge=1, le=255)
    region: ImageRegion


class ClassificationRequest(Contract):
    base_revision: int = Field(ge=0)
    model_id: str | None = None
    categories: list[str] = Field(min_length=2, max_length=16)
    objects: list[ClassificationObject] = Field(min_length=1, max_length=128)
    prompt: str = Field(default="", max_length=8000)

    @model_validator(mode="after")
    def validate_objects_and_categories(self) -> Self:
        if len({o.id for o in self.objects}) != len(self.objects):
            raise ValueError("Classification object IDs must be unique.")
        names = [name.strip().casefold() for name in self.categories]
        if any(not name or len(name) > 80 for name in names) or len(set(names)) != len(names):
            raise ValueError("Choose unique, nonempty category names up to 80 characters.")
        return self


class ObjectClassification(Contract):
    object_id: str
    category: str | None  # Null means the model abstained; never invent a category.


class ClassificationProposal(Record):
    project_id: str
    asset_id: str
    base_revision: int
    model_id: str
    request: ClassificationRequest
    results: list[ObjectClassification]


class AnnotateRequest(Contract):
    model_id: str | None = None
    label_ids: list[int] | None = None
    prompt: str = Field(default="", max_length=8000)
    slice: SliceScope | None = None
    all_slices: bool = False
    image_region: ImageRegion | None = None
    image_tiling: ImageTiling | None = None
    spatial_prompt: SpatialPrompt | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.image_region and self.image_tiling:
            raise ValueError("A selected region uses one crop; tiling is for the full image.")
        if (self.image_region or self.image_tiling) and (self.slice or self.all_slices):
            raise ValueError("Choose a 2D image region or a volume slice scope.")
        if self.all_slices and self.slice is None:
            raise ValueError(
                "All-slice annotation requires a slice plane, orientation, and window."
            )
        return self


class BatchAnnotateRequest(Contract):
    limit: int = Field(default=5, ge=1, le=100)
    model_id: str | None = None
    label_ids: list[int] = Field(default_factory=list, max_length=31)
    submit_for_review: bool = False
    prompt: str = Field(default="", max_length=16000)


class Proposal(Record):
    project_id: str
    asset_id: str
    base_revision: int
    model_ids: list[str]
    label_ids: list[int]
    mask_key: str
    prompt: str
    status: Literal["pending", "accepted", "rejected"] = "pending"
    slice: SliceScope | None = None
    all_slices: bool = False
    volume_plane: SliceScope | None = None
    image_region: ImageRegion | None = None
    image_tiling: ImageTiling | None = None
    spatial_prompt: SpatialPrompt | None = None


Pixel = Annotated[int, Field(strict=True, ge=0, le=255)]


class ReviewRequest(Contract):
    base_revision: int = Field(ge=0)
    proposal_id: str | None = None
    mask: list[JsonValue] | None = None
    covered_labels: list[int] = Field(min_length=1, max_length=32)
    reviewer: str = Field(default="", max_length=120)


class Annotation(Record):
    project_id: str
    asset_id: str
    revision: int
    mask_key: str
    covered_labels: list[int]
    reviewer: str
    proposal_id: str | None = None
    restored_from: str | None = None


class RestoreRequest(Contract):
    annotation_id: str
    base_revision: int = Field(ge=0)
    reviewer: str = Field(min_length=1, max_length=120)


class Sample(Contract):
    asset_id: str
    image_key: str
    mask_key: str
    revision: int
    group_id: str
    split: Split
    decision_id: str | None = None
    affine: list[list[float]] | None = None
    label_source: Literal["reviewed", "model_prediction"] = "reviewed"
    proposal_id: str | None = None
    model_ids: list[str] = Field(default_factory=list)


class TrainingSampleFilter(Contract):
    source_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)
    asset_ids: list[str] | None = Field(default=None, min_length=1, max_length=10000)
    limit: int | None = Field(default=None, ge=1, le=10000)


class TrainingSource(Contract):
    id: str
    name: str
    asset_ids: list[str]


class TrainingLabelPolicy(Contract):
    allow_unreviewed_training: bool = False
    note: str = Field(
        default="", max_length=1000, description="Optional note for this training run."
    )


class SnapshotRequest(TrainingLabelPolicy):
    sample_filter: TrainingSampleFilter = Field(default_factory=TrainingSampleFilter)
    label_ids: list[int] | None = None


class Snapshot(Record):
    sample_filter: TrainingSampleFilter = Field(default_factory=TrainingSampleFilter)
    model_split_id: str | None = None
    model_split_version: int | None = None
    evaluation_version_id: str | None = None
    project_id: str
    protocol_version: int
    labels: list[Label]
    samples: list[Sample]
    allow_unreviewed_training: bool = False
    authorized_by: str | None = None
    note: str = ""


class TrainingMode(StrEnum):
    SCRATCH = "scratch"
    FINE_TUNE = "fine_tune"
    CONTINUE = "continue"


class LearnerCreate(Contract):
    name: str = Field(min_length=1, max_length=120)
    recipe: str = Field(min_length=1, max_length=80)
    label_ids: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=32)
    inherit_targets: bool = False
    config: dict[str, JsonValue] = Field(default_factory=dict)
    initial_model_id: str | None = None


class Learner(LearnerCreate, Record):
    evaluation_set_id: str | None = None
    project_id: str
    protocol_version: int
    version: int = 0
    archived: bool = False


class LearnerUpdate(ModelUpdate):
    """Rename a training setup without changing its recommended settings."""


class RecipeInfo(Contract):
    id: str
    name: str
    description: str
    available: bool
    default_config: dict[str, JsonValue] = Field(default_factory=dict)
    demo_only: bool = False
    setup: str = ""
    supported_targets: list[str] | None = None
    target_class_ids: dict[str, int] = Field(default_factory=dict)
    documentation_url: str | None = None


class StartTraining(TrainingLabelPolicy):
    sample_filter: TrainingSampleFilter = Field(default_factory=TrainingSampleFilter)
    validation_percentage: int | None = Field(default=None, ge=1, le=50)
    evaluation_set_id: str | None = None
    evaluation_version_id: str | None = None
    config: dict[str, JsonValue] = Field(default_factory=dict)
    snapshot_id: str | None = None
    mode: TrainingMode = TrainingMode.SCRATCH
    parent_model_id: str | None = None
    label_ids: list[int] | None = Field(default=None, min_length=2, max_length=32)

    @model_validator(mode="after")
    def check_parent(self) -> Self:
        if self.snapshot_id and self.sample_filter != TrainingSampleFilter():
            raise ValueError("Sample filters require a new snapshot.")
        if self.validation_percentage is not None and (
            self.evaluation_set_id or self.evaluation_version_id or self.snapshot_id
        ):
            raise ValueError("Choose a model split or existing references, not both.")
        if self.evaluation_set_id and self.evaluation_version_id:
            raise ValueError("Choose an evaluation set or a specific reference version.")
        if (self.mode == TrainingMode.SCRATCH) == (self.parent_model_id is not None):
            raise ValueError("Scratch has no parent; fine-tune and continue require a checkpoint.")
        return self


class TrainRequest(Contract):
    snapshot_id: str
    recipe: str = "pixel-gaussian"
    config: dict[str, JsonValue] = Field(default_factory=dict)
    label_ids: list[int] | None = None
    learner_id: str | None = None
    mode: TrainingMode = TrainingMode.SCRATCH
    parent_model_id: str | None = None
    name: str = Field(default="Project segmenter", min_length=1, max_length=120)

    @model_validator(mode="after")
    def check_parent(self) -> Self:
        if (self.mode == TrainingMode.SCRATCH) == (self.parent_model_id is not None):
            raise ValueError("Scratch training has no parent; fine_tune/continue require a parent.")
        return self


class EvaluateRequest(Contract):
    evaluation_set_id: str | None = None
    evaluation_version_id: str | None = None
    snapshot_id: str | None = None
    candidate_id: str
    baseline_id: str
    label_ids: list[int] | None = Field(default=None, min_length=1)
    min_dice: float = Field(default=0.90, ge=0, le=1)
    min_improvement: float = Field(default=0.001, ge=0, le=1)

    @model_validator(mode="after")
    def check_reference(self) -> Self:
        if (
            sum(
                value is not None
                for value in (self.evaluation_set_id, self.evaluation_version_id, self.snapshot_id)
            )
            > 1
        ):
            raise ValueError("Choose one evaluation set, reference version or snapshot.")
        return self


class ModelMetrics(Contract):
    per_class: dict[int, float]
    mean_dice: float


class Evaluation(Record):
    evaluation_version_id: str | None = None
    project_id: str
    snapshot_id: str | None
    candidate_id: str
    baseline_id: str
    candidate: ModelMetrics
    baseline: ModelMetrics
    eligible_labels: list[int]
    min_dice: float
    min_improvement: float
    validation_assets: list[str]


class TrainingReport(Record):
    model_split_id: str | None = None
    model_split_version: int | None = None
    project_id: str
    training_job_id: str
    model_id: str
    snapshot_id: str
    evaluation_version_id: str | None = None
    labels: list[Label]
    validation_assets: list[str] = Field(default_factory=list)
    metrics: ModelMetrics | None = None
    per_class_iou: dict[int, float] = Field(default_factory=dict)
    initial_loss: float | None = None
    final_loss: float | None = None
    error: str | None = None


class PromoteRequest(Contract):
    evaluation_id: str
    label_ids: list[int] = Field(min_length=1)
    base_version: int = Field(ge=0)


class Promotion(Record):
    project_id: str
    evaluation_id: str
    model_id: str
    label_ids: list[int]
    previous_defaults: dict[int, str]
    project_version: int
    reverted: bool = False


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}
)


class Job(Record):
    project_id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0
    progress_message: str = ""
    request: dict[str, JsonValue]
    result: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None
    idempotency_key: str | None = None
    log_count: int = 0


class JobLog(Record):
    project_id: str
    job_id: str
    sequence: int
    level: Literal["info", "error"] = "info"
    message: str


class JobLogPage(Contract):
    entries: list[JobLog]
    next_cursor: int
    truncated: bool = False
    total_lines: int = 0


class SelectionRequest(Contract):
    strategy: Literal["random", "disagreement"] = "random"
    limit: int = Field(default=5, ge=1, le=100)
    model_ids: list[str] = Field(default_factory=list)
    seed: int = 42


class WorkItem(Contract):
    asset_id: str
    score: float
    reason: str


class AssistantContext(Contract):
    evaluation_set_id: str | None = None
    evaluation_version_id: str | None = None
    viewer_actions: list[str] = Field(default_factory=list, max_length=16)
    learner_id: str | None = None
    asset_id: str | None = None
    model_id: str | None = None
    baseline_id: str | None = None
    snapshot_id: str | None = None
    evaluation_id: str | None = None
    slice: SliceScope | None = None
    image_region: ImageRegion | None = None
    image_tiling: ImageTiling | None = None
    spatial_prompt: SpatialPrompt | None = None
    spatial_objects: list[SpatialObject] = Field(default_factory=list, max_length=128)
    label_ids: list[int] | None = None
    classification: ClassificationRequest | None = None
    base_revision: int | None = Field(default=None, ge=0)


class AssistantRequest(Contract):
    message: str = Field(min_length=1, max_length=8000)
    context: AssistantContext = Field(default_factory=AssistantContext)
    conversation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    continue_tool: str | None = Field(default=None, max_length=200)
    request_id: str = Field(default_factory=new_id, pattern=r"^[a-f0-9]{32}$")


class AssistantReply(Contract):
    assistant: str
    message: str
    job_id: str | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)
    conversation_id: str | None = None
    tools: list[str] = Field(default_factory=list)


class User(Record):
    username: str
    is_admin: bool = False
    active: bool = True


class Role(StrEnum):
    MANAGER = "manager"
    ANNOTATOR = "annotator"
    REVIEWER = "reviewer"


class Membership(Record):
    project_id: str
    user_id: str
    roles: list[Role] = Field(min_length=1)


class CompleteReview(Contract):
    base_revision: int = Field(ge=1)
    covered_labels: list[int] = Field(min_length=1, max_length=32)
    comment: str = Field(default="", max_length=4000)


class ReviewDecision(Record):
    project_id: str
    asset_id: str
    annotation_id: str
    revision: int
    reviewer_id: str
    verdict: Literal["accepted", "changes_requested", "pending"]
    comment: str = Field(default="", max_length=4000)


class ReviewTarget(Contract):
    annotation_id: str
    decision_id: str | None = None


class BatchReviewRequest(Contract):
    items: list[ReviewTarget] = Field(min_length=1, max_length=10000)
    verdict: Literal["accepted", "changes_requested", "pending"]
    comment: str = Field(default="", max_length=4000)


class DecisionRequest(Contract):
    verdict: Literal["accepted", "changes_requested", "pending"]
    comment: str = Field(default="", max_length=4000)


class Credential(Record):
    project_id: str
    name: str
