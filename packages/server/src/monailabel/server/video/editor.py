"""Durable CVAT task bindings; opening a task never overwrites a saved draft."""

import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import httpx
from filelock import FileLock, Timeout

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import Job, Project, User
from monailabel.core.video import (
    TrackAnnotation,
    TrackSubmission,
    VideoAsset,
    VideoEditorRequest,
)
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.secrets import EncryptedCredential, Secrets
from monailabel.server.storage import Store
from monailabel.server.video.assets import Videos
from monailabel.server.video.models import ManagedCvatRuntime, VideoEditor
from monailabel.viewers.cvat import CvatClient, decode_tracks, validate_url
from monailabel.viewers.cvat_runtime import CvatManager


class VideoEditors:
    def __init__(self, store: Store, videos: Videos, jobs: Jobs, credentials: Secrets):
        self.store, self.videos, self.jobs = store, videos, jobs
        self.credentials = credentials
        self.setup_lock = Lock()
        runtime = next(iter(store.list(ManagedCvatRuntime)), None)
        if runtime is None:
            runtime = ManagedCvatRuntime()
            with store.transaction() as session:
                session.insert(runtime)
        namespace = runtime.namespace
        root = Path(os.environ.get("MONAILABEL_CACHE_DIR", str(store.path.parent / ".cache")))
        self.manager = CvatManager(root, namespace)
        url = os.environ.get("MONAILABEL_CVAT_URL", "")
        token = os.environ.get("MONAILABEL_CVAT_TOKEN", "")
        self.public_url = (
            validate_url(os.environ.get("MONAILABEL_CVAT_PUBLIC_URL") or url) if url else ""
        )
        self.managed = not url and not token
        self.client = CvatClient(url, token) if url and token else None

    def close(self) -> None:
        if self.client:
            self.client.close()

    def configured(self, context: JobContext | None = None) -> CvatClient:
        if self.managed and not self.client:
            with self.setup_lock:
                if not self.client:
                    identifier = "managed-cvat-service"
                    stored = next(
                        (r for r in self.store.list(EncryptedCredential) if r.id == identifier),
                        None,
                    )
                    if stored is None:
                        stored = EncryptedCredential(
                            id=identifier,
                            ciphertext=self.credentials.cipher.encrypt(
                                secrets.token_urlsafe(48).encode()
                            ).decode(),
                        )
                        with self.store.transaction() as session:
                            session.insert(stored)
                    password = self.credentials.cipher.decrypt(stored.ciphertext.encode()).decode()
                    username = "monailabel_service"
                    url = self.manager.ensure(
                        username,
                        password,
                        lambda message: context.progress(0.02, message) if context else None,
                    )
                    try:
                        with httpx.Client(timeout=30) as http:
                            response = http.post(
                                url + "/api/auth/login",
                                json={"username": username, "password": password},
                            )
                            response.raise_for_status()
                            token = response.json()["key"]
                    except (httpx.HTTPError, KeyError, ValueError) as exc:
                        raise DomainError(
                            "Could not sign in to the managed CVAT service. "
                            "Its saved drafts are preserved.",
                            status=503,
                        ) from exc
                    self.client = CvatClient(url, token)
        if not self.client:
            raise DomainError(
                "Configure MONAILABEL_CVAT_URL and MONAILABEL_CVAT_TOKEN to open CVAT. "
                "See docs/video.md for the local service setup.",
                status=503,
            )
        return self.client

    @contextmanager
    def lock(self, asset_id: str) -> Iterator[None]:
        # Asset IDs come from saved records, never a user-supplied file path.
        try:
            with FileLock(str(self.store.path.parent / f"video-{asset_id}.lock"), timeout=0):
                yield
        except Timeout as exc:
            raise Conflict(
                "A CVAT operation is already running for this clip. Retry when it finishes."
            ) from exc

    def start(self, video_id: str, request: VideoEditorRequest) -> Job:
        if not self.managed:
            self.configured()
        asset = self.store.get(VideoAsset, video_id)
        if asset.revision != request.base_revision:
            raise Conflict("The submitted video changed. Refresh before opening the editor.")
        if request.mode == "review" and not asset.annotation_id:
            raise DomainError("Submit tracks before opening a review task.")
        project = self.store.get(Project, asset.project_id)
        if not any(label.id for label in project.labels):
            raise DomainError("Add instrument labels when importing the clip before opening CVAT.")
        return self.jobs.submit(
            "video_editor",
            asset.project_id,
            {"asset_id": asset.id, **request.model_dump(mode="json")},
            lambda context: self.prepare(asset, project, request, context),
        )

    def prepare(
        self, asset: VideoAsset, project: Project, request: VideoEditorRequest, context: JobContext
    ) -> Outcome:
        client = self.configured(context)
        with self.lock(asset.id):
            if self.store.get(VideoAsset, asset.id).revision != request.base_revision:
                raise Conflict("The submitted video changed. Refresh before opening the editor.")
            for editor in reversed(self.store.list(VideoEditor, asset.project_id)):
                if (
                    editor.asset_id == asset.id
                    and editor.base_revision == request.base_revision
                    and editor.mode == request.mode
                    and editor.server_url == client.url
                    and editor.ready
                    and not editor.submitted_annotation_id
                ):
                    client.enable_polygon_labels(editor.task_id)
                    return self.outcome(editor)
            context.progress(0.1, "Creating a CVAT task for this video revision.")
            task_id = client.create_task(
                f"{project.name} / {asset.name} / r{asset.revision} / {request.mode}",
                project.labels,
            )
            editor = VideoEditor(
                project_id=asset.project_id,
                asset_id=asset.id,
                base_revision=asset.revision,
                mode=request.mode,
                server_url=client.url,
                task_id=task_id,
            )
            with self.store.transaction() as session:
                session.insert(editor)
            request_id = client.upload(
                task_id, self.videos.artifacts.path(asset.source_key), asset.name
            )

            def pause() -> None:
                context.progress(0.4, "CVAT is decoding the original video frames.")
                time.sleep(1)

            client.wait(request_id, pause)
            client.check_frames(task_id, asset)
            labels = client.label_map(task_id, project.labels)
            document = self.videos.document(asset)
            context.progress(0.8, "Opening the annotation task.")
            identities = client.seed(task_id, document, labels)
            roundtrip = decode_tracks(
                client.annotations(task_id),
                {v: k for k, v in labels.items()},
                identities,
                editor.id,
            )
            if roundtrip != document:
                raise DomainError("CVAT did not preserve this annotation revision exactly.")
            editor = editor.model_copy(
                update={
                    "ready": True,
                    "job_id": client.job_id(task_id),
                    "label_map": labels,
                    "track_ids": identities,
                }
            )
            with self.store.transaction() as session:
                session.update(editor)
            return self.outcome(editor)

    def outcome(self, editor: VideoEditor) -> Outcome:
        return Outcome(
            result={
                "editor_id": editor.id,
                "video_id": editor.asset_id,
                "url": (
                    f"/cvat/editor/{editor.id}"
                    if self.managed
                    else f"{self.public_url}/tasks/{editor.task_id}/jobs/{editor.job_id}"
                ),
            }
        )

    def submit(self, video_id: str, editor_id: str, user: User) -> TrackAnnotation:
        client = self.configured()
        asset = self.store.get(VideoAsset, video_id)
        with self.lock(asset.id):
            editor = self.store.get(VideoEditor, editor_id)
            if editor.asset_id != asset.id or editor.server_url != client.url or not editor.ready:
                raise DomainError("Choose a ready CVAT task belonging to this video and service.")
            if editor.submitted_annotation_id:
                return self.store.get(TrackAnnotation, editor.submitted_annotation_id)
            if editor.base_revision != asset.revision:
                raise Conflict("A newer video revision exists. Your CVAT draft is preserved.")
            client.check_frames(editor.task_id, asset)
            document = decode_tracks(
                client.annotations(editor.task_id),
                {v: k for k, v in editor.label_map.items()},
                editor.track_ids,
                editor.id,
            )
            # Publish the revision and the consumed editor receipt in one transaction.
            return self.videos.submit(
                asset.id,
                TrackSubmission(base_revision=editor.base_revision, document=document),
                user,
                editor=editor,
            )
