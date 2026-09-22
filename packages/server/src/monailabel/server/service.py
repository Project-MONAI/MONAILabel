"""Composition root: application services share storage and execution ports."""

from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock

from monailabel.core.chat import ChatProvider
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.annotation import Annotations
from monailabel.server.assistants import Assistants
from monailabel.server.auth import Auth
from monailabel.server.batch_annotation import BatchAnnotations
from monailabel.server.classification import Classifications
from monailabel.server.coordinator_runtime import CoordinatorRuntime
from monailabel.server.data import Datasets
from monailabel.server.dataset_downloads import dataset_cache_dir
from monailabel.server.dataset_templates import DatasetTemplates
from monailabel.server.deletion import Deletion, cleanup_storage
from monailabel.server.desktops.sessions import DesktopSessions
from monailabel.server.dicom.connections import DicomConnections
from monailabel.server.dicom.service import Dicom
from monailabel.server.evaluation_sets import EvaluationSets
from monailabel.server.jobs import Jobs
from monailabel.server.learning import Learning
from monailabel.server.models import Models
from monailabel.server.models.catalog import ModelCatalogs
from monailabel.server.models.presets import Presets
from monailabel.server.reference_imports import ReferenceImports
from monailabel.server.regions import Regions
from monailabel.server.review_units.service import ReviewUnits
from monailabel.server.reviews import Reviews
from monailabel.server.scoring import ValidationScorer
from monailabel.server.secrets import Secrets
from monailabel.server.selection import Selection
from monailabel.server.storage import Artifacts, Store
from monailabel.server.training_reports import TrainingReports
from monailabel.server.video.assets import Videos
from monailabel.server.video.editor import VideoEditors
from monailabel.server.video.tracking import VideoTracking


class Services:
    def __init__(
        self,
        data_dir: Path,
        *,
        chat_provider: ChatProvider | None = None,
        coordinator: CoordinatorConfig | None = None,
    ):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(data_dir / "server.lock"), timeout=0)
        with ExitStack() as cleanup:
            self.lock.acquire()
            cleanup.callback(self.lock.release)
            self.store = Store(data_dir / "monailabel.sqlite3")
            self.auth = Auth(self.store)
            self.desktops = DesktopSessions(self.store, self.auth, data_dir)
            cleanup.callback(self.desktops.close)
            self.secrets = Secrets(self.store, data_dir)
            self.reviews = Reviews(self.store)
            self.artifacts = Artifacts(data_dir / "artifacts")
            cleanup_storage(self.store, self.artifacts)
            ReviewUnits(self.store, self.artifacts).migrate_videos()
            self.deletion = Deletion(self.store)
            # Registered before workers so their clients close after jobs have finished.
            editor_cleanup = cleanup.enter_context(ExitStack())
            self.jobs = Jobs(self.store)
            cleanup.callback(self.jobs.close)
            self.models = Models(self.store, self.artifacts, self.secrets.resolve)
            self.presets = Presets(self.store, self.secrets.resolve)
            self.model_catalogs = ModelCatalogs(self.models, self.secrets.resolve, self.presets)
            from monailabel.core.models import Project

            for project in self.store.list(Project):
                self.presets.ensure(project.id)
            self.evaluation_sets = EvaluationSets(self.store)
            self.datasets = Datasets(self.store, self.artifacts)
            self.videos = Videos(self.store, self.artifacts)
            self.video_tracking = VideoTracking(self.store, self.artifacts, self.jobs, self.models)
            self.video_editors = VideoEditors(self.store, self.videos, self.jobs, self.secrets)
            editor_cleanup.callback(self.video_editors.close)
            self.reference_imports = ReferenceImports(self.store, self.artifacts)
            self.dicom = Dicom(self.store, self.artifacts, self.jobs)
            self.dicom_connections = DicomConnections(
                self.store, self.artifacts, self.jobs, self.secrets
            )
            self.regions = Regions(self.store, self.artifacts, self.models, self.jobs)
            self.annotations = Annotations(self.store, self.artifacts, self.models, self.jobs)
            self.batch_annotations = BatchAnnotations(self.store, self.annotations, self.jobs)
            self.dataset_templates = DatasetTemplates(
                self.store,
                self.datasets,
                self.annotations,
                self.jobs,
                dataset_cache_dir(data_dir),
                references=self.reference_imports,
                videos=self.videos,
                legacy_cache=data_dir / "dataset-downloads",
            )
            self.classifications = Classifications(
                self.store, self.artifacts, self.models, self.jobs
            )
            self.scorer = ValidationScorer(self.store, self.artifacts, self.models)
            self.training_reports = TrainingReports(
                self.store, self.artifacts, self.scorer, self.jobs
            )
            self.learning = Learning(
                self.store,
                self.artifacts,
                self.models,
                self.jobs,
                self.datasets,
                self.scorer,
                self.training_reports,
            )
            self.selection = Selection(self.store, self.artifacts, self.models, self.jobs)
            self.coordinator: CoordinatorRuntime | None = None
            if chat_provider is None:
                self.coordinator = CoordinatorRuntime(coordinator or CoordinatorConfig.from_env())
                chat_provider = self.coordinator
            self.assistants = Assistants(self, chat_provider)
            if self.coordinator:
                cleanup.callback(self.coordinator.close)
                self.coordinator.start()
            self._cleanup = cleanup.pop_all()

    def close(self) -> None:
        self._cleanup.close()
