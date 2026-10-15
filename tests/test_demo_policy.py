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

"""Visitor capabilities hold at HTTP, tool-service, and background-job boundaries."""

import hashlib
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from monailabel.core.dataset_templates import DatasetTemplateImport
from monailabel.core.dicom import DicomConnection, DicomImportSelection
from monailabel.core.errors import DomainError
from monailabel.core.models import (
    Asset,
    ImageImport,
    ImageMetadata,
    Job,
    Label,
    ModelRecord,
    ModelRegister,
    Project,
    Role,
    User,
)
from monailabel.core.ports import Prediction
from monailabel.core.preparation import prepared_models_only
from monailabel.core.reference_imports import LabelImportRequest
from monailabel.core.video import VideoAsset, VideoImport
from monailabel.server.dataset_downloads import Source
from monailabel.server.demo_context import acting_as
from monailabel.server.demo_policy import prepared_import, require_prepared_import
from monailabel.server.jobs import Outcome
from monailabel.server.secrets import Credential, EncryptedCredential


@pytest.fixture
def visit(http, client, monkeypatch):
    # Invited users must not regain capabilities when public signup or the
    # optional resource-expiry sweeper is disabled.
    for suffix in ("SELF_REGISTER", "LIFECYCLE", "CACHE_ONLY", "TRAINING_PREVIEW"):
        monkeypatch.setenv("MONAILABEL_DEMO_" + suffix, "0")
    service = http.app.state.services
    project = client.post("/api/projects", {"name": "Prepared visitor project"})
    visitor = service.auth.create_user("visitor", "test-password-1234", demo_minutes=60)
    service.auth.add_member(project["id"], visitor.id, [Role.MANAGER])
    http.post(
        "/api/auth/login", json={"username": "visitor", "password": "test-password-1234"}
    ).raise_for_status()
    return service, project["id"], visitor.id


def denied(response):
    assert response.status_code == 403, response.text
    assert response.json()["code"] == "demo_restricted"


def prepared_asset(service, project, *, volume=False):
    asset = Asset(
        project_id=project,
        name="Prepared synthetic sample",
        group_id="synthetic-case",
        split="pool",
        image_key="unused-in-launch-preflight",
        spatial_shape=[4, 4, 4] if volume else [4, 4],
        kind="volume3d" if volume else "image2d",
    )
    with service.store.transaction() as session:
        session.insert(asset)
    return asset


@pytest.mark.parametrize("viewer", ["slicer", "qupath", "ohif"])
def test_visitor_viewer_cache_miss_fails_before_queue_or_credentials(
    http, visit, monkeypatch, viewer
):
    service, project, user = visit
    asset = prepared_asset(service, project, volume=viewer != "qupath")
    failure = Mock(side_effect=DomainError("Test viewer not prepared", status=503))
    monkeypatch.setattr(service.desktops.runtime, "preflight", failure)
    monkeypatch.setattr("monailabel.viewers.ohif.OhifManager.ensure", failure)
    credentials = Mock(side_effect=AssertionError("No new credential allowed"))
    monkeypatch.setattr(service.auth, "issue", credentials)
    before = service.store.list(Job)
    response = http.post(f"/api/assets/{asset.id}/viewer?name={viewer}")
    assert response.status_code == 503 and "not prepared" in response.text
    assert service.store.list(Job) == before
    credentials.assert_not_called()
    if viewer == "ohif":
        assert failure.call_args.kwargs["download"] is False
    else:
        # Direct service callers get the same check before a session is allocated.
        with acting_as(user), pytest.raises(DomainError, match="not prepared"):
            service.desktops.open(
                asset,
                service.store.get(User, user),
                viewer,
                "annotation",
                "http://127.0.0.1:8000",
                SimpleNamespace(progress=lambda *args: None),
            )
        credentials.assert_not_called()


def test_leased_localhost_uses_managed_desktop_without_expiry_sweeper(
    http, client, visit, monkeypatch
):
    service, project, user = visit
    asset = prepared_asset(service, project)
    monkeypatch.setattr("monailabel.server.console_api.local_desktop", lambda request: True)
    native = Mock(side_effect=AssertionError("A visitor must never launch on the host desktop"))
    monkeypatch.setattr("monailabel.viewers.manager.ViewerManager.ensure", native)
    preflight = Mock()
    monkeypatch.setattr(service.desktops.runtime, "preflight", preflight)
    opened = Mock(return_value=SimpleNamespace(id="test-managed-desktop"))
    monkeypatch.setattr(service.desktops, "open", opened)
    response = http.post(f"/api/assets/{asset.id}/viewer?name=qupath&target=auto")
    assert response.status_code == 202, response.text
    result = client.wait(response.json()["id"])
    assert result["url"] == "/desktop/test-managed-desktop"
    assert opened.call_args.args[1].id == user
    preflight.assert_called_once_with("qupath")
    native.assert_not_called()


@pytest.mark.parametrize("managed", [False, True])
def test_visitor_cvat_cannot_configure_or_start_unprepared_runtime(
    http, visit, monkeypatch, managed
):
    service, project, user = visit
    editor = service.video_editors
    monkeypatch.setattr(editor, "managed", managed)
    video = VideoAsset(
        project_id=project,
        name="Prepared synthetic clip",
        group_id="synthetic-case",
        split="pool",
        source_key="unused",
        metadata_key="unused",
        width=4,
        height=4,
        frames=1,
        duration=1,
    )
    with service.store.transaction() as session:
        session.insert(video)
    before = service.store.list(EncryptedCredential)
    failure = Mock(side_effect=DomainError("Test CVAT not prepared", status=503))
    monkeypatch.setattr(editor.manager, "prepared", failure)
    ensure = Mock(side_effect=AssertionError("No service installation or startup allowed"))
    monkeypatch.setattr(editor.manager, "ensure", ensure)
    response = http.post(f"/api/videos/{video.id}/editor", json={"base_revision": 0})
    assert response.status_code == (503 if managed else 403), response.text
    assert not service.store.list(Job)
    with acting_as(user), pytest.raises(DomainError):
        editor.configured()
    assert service.store.list(EncryptedCredential) == before
    ensure.assert_not_called()


def test_http_rejects_configuration_and_uploads_before_side_effects(http, visit, monkeypatch):
    service, project, _ = visit
    prefix = f"/api/projects/{project}"
    forbidden = Mock(side_effect=AssertionError("No network, decoding, or buffering allowed"))
    monkeypatch.setattr("monailabel.server.api.read_upload", forbidden)
    monkeypatch.setattr("monailabel.server.data.decode_image", forbidden)
    monkeypatch.setattr("monailabel.server.models.catalog.discover", forbidden)
    monkeypatch.setattr("monailabel.server.dicom.connections.DicomwebClient", forbidden)
    monkeypatch.setattr(service.dicom, "request", forbidden)
    before = {kind: service.store.list(kind) for kind in (Asset, ModelRecord, Credential, Job)}
    artifacts = set(service.artifacts.root.rglob("*"))
    for route, body in (
        ("/credentials", {"name": "No credentials", "api_key": "disposable-test-value"}),
        ("/models", {"name": "External model", "provider": "http-mask", "label_ids": [0, 1]}),
        ("/model-catalog", {"service": "openai"}),
        ("/model-import", {"service": "openai", "model": "unavailable-fixture"}),
        ("/dicom-connections", {"name": "External", "url": "http://127.0.0.1:9"}),
        ("/dicom-series", {"series_uid": "1.2.3"}),
        ("/assets", {"name": "invalid.png", "image_base64": "not base64"}),
    ):
        denied(http.post(prefix + route, json=body))
    for route in (
        "/assets/upload?name=invalid.png",
        "/videos/upload?name=invalid.mp4&group_id=fixture",
        "/label-imports",
        "/evaluation-imports",
    ):
        denied(http.post(prefix + route, content=b"not an image, video, or multipart form"))
    forbidden.assert_not_called()
    assert {kind: service.store.list(kind) for kind in before} == before
    assert set(service.artifacts.root.rglob("*")) == artifacts


def test_services_do_not_depend_on_http_guards(visit, tmp_path):
    service, project, user = visit
    calls = (
        lambda: service.secrets.save(project, "test", "not-a-real-secret"),
        lambda: service.models.register(project, ModelRegister(name="x", provider="http-mask")),
        lambda: service.datasets.import_image(
            project, ImageImport(name="x.png", image_base64="invalid")
        ),
        lambda: service.datasets.import_content(project, ImageMetadata(name="x.png"), b"bad"),
        lambda: service.reference_imports.import_pair(
            project,
            LabelImportRequest(labels={1: "region"}),
            "x.png",
            b"bad",
            "y.png",
            b"bad",
            user,
        ),
        lambda: service.videos.import_file(
            project, VideoImport(name="x.mp4", group_id="case"), tmp_path / "missing"
        ),
        lambda: service.dicom_connections.connect(project, "test", "invalid", "none"),
        lambda: service.dicom_connections.import_selection(
            project,
            "missing",
            DicomImportSelection(series=[{"study_uid": "1.2", "series_uid": "1.3"}]),
        ),
        lambda: service.dicom.import_series(project, "1.2.3"),
    )
    with acting_as(user):
        for call in calls:
            with pytest.raises(DomainError) as error:
                call()
            assert error.value.status == 403 and error.value.code == "demo_restricted"


def test_prepared_import_context_is_actor_project_bound_and_exception_safe(visit):
    service, project, user = visit
    other = service.auth.create_user("another", "test-password-1234", demo_minutes=60)

    def blocked(identifier, destination):
        with acting_as(identifier), pytest.raises(DomainError, match="managed by the demo host"):
            require_prepared_import(service.store, destination)

    with acting_as(user):
        blocked(user, project)
        with pytest.raises(RuntimeError), prepared_import(project):
            require_prepared_import(service.store, project)
            blocked(user, "another-project")
            blocked(other.id, project)
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(blocked, user, project).result(timeout=5)
            raise RuntimeError("Import failed")
        blocked(user, project)


def test_visitor_cannot_use_existing_hosted_connections_or_models(http, visit, monkeypatch):
    service, project, user = visit
    hosted = ModelRecord(
        project_id=project, name="Host configuration", provider="http-mask", label_ids=[0, 1]
    )
    connection = DicomConnection(project_id=project, name="Host PACS", url="http://127.0.0.1:9")
    with service.store.transaction() as session:
        session.insert(hosted)
        session.insert(connection)
    forbidden = Mock(side_effect=AssertionError("Never call a hosted provider"))
    monkeypatch.setattr(service.models.providers["http-mask"], "predict", forbidden)
    with acting_as(user):
        assert hosted.id not in {m.id for m in service.models.available(project)}
        for call in (
            lambda: service.models.get(project, hosted.id),
            lambda: service.models.predict(
                service.store.get(Project, project), hosted, np.zeros((2, 2, 1))
            ),
            lambda: service.dicom_connections.client(connection),
        ):
            with pytest.raises(DomainError) as error:
                call()
            assert error.value.code == "demo_restricted"
        for kind in ("train", "classify", "roi", "bounding_box"):
            with pytest.raises(DomainError) as error:
                service.jobs.submit(kind, project, {}, lambda _: Outcome({}))
            assert error.value.code == "demo_restricted"
    assert service.store.list(Job) == []
    assert http.get(f"/api/projects/{project}/model-services").json() == []
    forbidden.assert_not_called()


def test_visitor_presets_skip_hosted_discovery(http, visit, monkeypatch, tmp_path):
    service, project, user = visit
    monkeypatch.setenv("MONAILABEL_MODELS_DIR", str(tmp_path / "missing-models"))
    service.presets.enabled = True
    forbidden = Mock(side_effect=AssertionError("Visitor reads must not discover hosted models"))
    monkeypatch.setattr(service.presets, "manual_available", forbidden)
    monkeypatch.setattr("monailabel.server.models.presets.discover", forbidden)
    with acting_as(user):
        service.presets.ensure(project)
        models = service.models.available(project)
    assert {m.provider for m in service.store.list(ModelRecord, project)} >= {
        "vista3d",
        "totalsegmentator-ct",
        "nninteractive",
    }
    assert not models  # Metadata exists, but visitors cannot select unprepared weights/runtime.
    assert all(m.provider in service.models.LOCAL_PROVIDERS for m in models)
    forbidden.assert_not_called()


def test_visitor_model_cache_miss_is_rejected_before_job_admission(
    http, visit, monkeypatch, tmp_path
):
    service, project, user = visit
    root = tmp_path / "missing-models"
    monkeypatch.setenv("MONAILABEL_MODELS_DIR", str(root))
    asset = prepared_asset(service, project, volume=True)
    model = ModelRecord(
        project_id=project,
        name="Unprepared VISTA3D",
        provider="vista3d",
        label_ids=[0, 1],
        read_only=True,
    )
    with service.store.transaction() as session:
        current = session.get(Project, project)
        session.update(
            current.model_copy(update={"labels": current.labels + [Label(id=1, name="spleen")]})
        )
        session.insert(model)
    before = service.store.list(Job)
    response = http.post(f"/api/assets/{asset.id}/annotate", json={"model_id": model.id})
    assert response.status_code == 503 and response.json()["code"] == "not_prepared"
    assert service.store.list(Job) == before
    assert not root.exists()
    with acting_as(user), pytest.raises(DomainError, match="not prepared"):
        service.models.get(project, model.id)


def test_model_execution_rechecks_visitor_policy_and_restores_context(visit, monkeypatch):
    service, project, user = visit
    model = ModelRecord(
        project_id=project, name="Prepared baseline", provider="threshold", label_ids=[0]
    )

    def predict(image, labels, prompt, selected):
        assert prepared_models_only()
        return Prediction(np.zeros(image.shape[:-1], np.uint8))

    monkeypatch.setattr(service.models.providers["threshold"], "predict", predict)
    assert not prepared_models_only()
    with acting_as(user):
        assert not prepared_models_only()  # Direct caller supplies identity, not a bypass flag.
        result = service.models.predict(
            service.store.get(Project, project), model, np.zeros((2, 3, 1))
        )
        assert result.shape == (2, 3)
    assert not prepared_models_only()


def test_background_job_has_prepared_only_policy_without_lifecycle_switch(visit, client):
    service, project, user = visit

    def work(context):
        assert prepared_models_only()
        # Internal maintenance identity changes cannot restore download capability.
        with acting_as(None):
            assert prepared_models_only()
        return Outcome({"prepared_only": True})

    with acting_as(user):
        job = service.jobs.submit("annotate", project, {}, work)
    assert client.wait(job.id) == {"prepared_only": True}
    assert not prepared_models_only()


@pytest.fixture
def prepared_source(visit, monkeypatch):
    service, _, _ = visit
    content = io.BytesIO()
    Image.fromarray(np.zeros((8, 9, 3), dtype=np.uint8)).save(content, format="PNG")
    payload = content.getvalue()
    source = Source(
        id="fixture",
        name="Synthetic prepared image",
        category="Pathology",
        description="Disposable test image",
        source_url="https://datasets.invalid/",
        license="Test",
        url="https://datasets.invalid/prepared.png",
        format="image",
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        download_bytes=len(payload),
    )
    downloads = service.dataset_templates.downloads
    downloads.root.mkdir(parents=True, exist_ok=True)
    downloads.path(source).write_bytes(payload)
    downloads._remember(source)
    monkeypatch.setattr("monailabel.server.dataset_templates.sources", lambda: [source])
    monkeypatch.setattr(httpx, "stream", Mock(side_effect=AssertionError("No download")))
    return downloads, source


def test_invited_cached_template_and_annotation_remain_available(
    http, client, visit, prepared_source
):
    service, project, user = visit
    prefix = f"/api/projects/{project}"
    assert [x["id"] for x in client.get(prefix + "/dataset-templates")] == ["fixture"]
    job = client.post(prefix + "/dataset-imports", {"template_id": "fixture", "limit": 1})
    result = client.wait(job["id"])
    asset = client.get("/api/assets/" + result["asset_ids"][0])
    assert service.store.get(Job, job["id"]).user_id == user
    mask = np.zeros(asset["spatial_shape"], dtype=np.uint8)
    response = http.post(
        f"/api/assets/{asset['id']}/review-mask",
        params={"base_revision": 0, "covered_labels": [0]},
        content=mask.tobytes(),
    )
    assert response.status_code == 201, response.text
    assert client.get("/api/assets/" + asset["id"])["revision"] == 1
    denied(http.post(prefix + "/assets", json={"name": "x.png", "image_base64": "invalid"}))
    httpx.stream.assert_not_called()


@pytest.mark.parametrize("limit", [6, None])
def test_invited_import_limit_rejects_before_queue(http, visit, prepared_source, limit):
    service, project, _ = visit
    response = http.post(
        f"/api/projects/{project}/dataset-imports", json={"template_id": "fixture", "limit": limit}
    )
    assert response.status_code == 413 and response.json()["code"] == "demo_import_limit"
    assert service.store.list(Job) == []


def test_cached_reference_import_preserves_labels_geometry_and_review(client, visit, monkeypatch):
    service, project, _ = visit
    image = np.zeros((3, 4, 5), dtype=np.int16)
    labels = np.ones(image.shape, dtype=np.uint8)
    affine = np.diag([-1.2, 1.5, 2.0, 1.0])
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, content in {
            "Task09_Spleen/dataset.json": json.dumps(
                {"labels": {"0": "background", "1": "spleen"}}
            ).encode(),
            "Task09_Spleen/imagesTr/case.nii": nib.Nifti1Image(image, affine).to_bytes(),
            "Task09_Spleen/labelsTr/case.nii": nib.Nifti1Image(labels, affine).to_bytes(),
        }.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    source = Source(
        id="Task09_Spleen",
        name="Synthetic labeled volume",
        category="Radiology",
        description="Disposable test volume",
        source_url="https://datasets.invalid/",
        license="Test",
        url="https://datasets.invalid/fixture.tar",
        format="msd",
        has_masks=True,
        checksum="sha256:" + hashlib.sha256(payload.getvalue()).hexdigest(),
    )
    downloads = service.dataset_templates.downloads
    downloads.root.mkdir(parents=True, exist_ok=True)
    downloads.path(source).write_bytes(payload.getvalue())
    downloads._remember(source)
    monkeypatch.setattr("monailabel.server.dataset_templates.sources", lambda: [source])
    prefix = f"/api/projects/{project}"
    result = client.wait(
        client.post(
            prefix + "/dataset-imports",
            {"template_id": source.id, "limit": 1, "include_masks": True, "split": "train"},
        )["id"]
    )
    assert not result["failed"]
    assert len(result["annotation_ids"]) == 1
    asset = service.store.get(Asset, result["asset_ids"][0])
    np.testing.assert_allclose(asset.affine, affine)
    assert asset.revision == 1 and asset.spatial_shape == list(image.shape)
    assert client.get(prefix + "/decisions") == []  # Imported training masks still need review.


def test_invited_cache_miss_never_downloads_without_cache_only_flag(http, visit, prepared_source):
    service, project, user = visit
    downloads, source = prepared_source
    downloads.path(source).write_bytes(b"changed after verification")
    with acting_as(user):
        assert service.dataset_templates.catalog() == []
        with pytest.raises(DomainError, match="not prepared"):
            service.dataset_templates.start(
                project, DatasetTemplateImport(template_id=source.id), user
            )
        # Also fail if the cache disappears between admission and worker execution.
        with pytest.raises(DomainError, match="not prepared"):
            downloads.fetch(source, SimpleNamespace(store=service.store, progress=lambda *_: None))
    assert service.store.list(Job) == []
    httpx.stream.assert_not_called()


def test_ordinary_manager_capabilities_unchanged(http, visit):
    service, project, _ = visit
    manager = service.auth.create_user("manager", "test-password-1234")
    service.auth.add_member(project, manager.id, [Role.MANAGER])
    http.post(
        "/api/auth/login", json={"username": "manager", "password": "test-password-1234"}
    ).raise_for_status()
    response = http.post(
        f"/api/projects/{project}/credentials", json={"name": "test", "api_key": "fixture-only"}
    )
    assert response.status_code == 201, response.text
    with acting_as(manager.id):
        require_prepared_import(service.store, project)


def test_fixed_synthetic_demo_remains_available_to_visitor(client, visit):
    setup = client.post("/api/demo")
    assert len(client.get(f"/api/projects/{setup['project_id']}/assets")) == 12
