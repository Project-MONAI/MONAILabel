import copy
import io
import json
import shutil
import subprocess
import time

import httpx
import numpy as np
import pytest
from PIL import Image

from monailabel.core.errors import DomainError
from monailabel.core.ports import VideoTrackingResult
from monailabel.core.video import TrackDocument, VideoAsset
from monailabel.viewers.cvat import CvatClient, decode_tracks, encode_tracks


@pytest.fixture(autouse=True)
def isolated_cvat(monkeypatch):
    for name in ("MONAILABEL_CVAT_URL", "MONAILABEL_CVAT_PUBLIC_URL", "MONAILABEL_CVAT_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clip(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for the video decoding integration test")
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-frames:v",
            "6",
            "-vf",
            "setpts=if(lt(N\\,3)\\,N/(10*TB)\\,(N+2)/(10*TB))",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path.read_bytes()


@pytest.fixture
def video(http, client, clip):
    project = client.post("/api/projects", {"name": "Video regression"})
    response = http.post(
        f"/api/projects/{project['id']}/videos/upload",
        params={"name": "clip.mp4", "group_id": "procedure-1", "labels": ["Grasper", "Scissors"]},
        content=clip,
    )
    assert response.status_code == 201, response.text
    return response.json()


def document():
    return {
        "tracks": [
            {
                "id": "instrument-a",
                "label_id": 1,
                "keyframes": [
                    {"frame": 0, "box": [1.25, 2, 20, 30]},
                    {"frame": 2, "box": [2, 3, 21, 31], "occluded": True},
                    {"frame": 4, "box": [3, 4, 22, 32], "outside": True},
                    {"frame": 5, "box": [4, 5, 23, 33]},
                ],
            }
        ]
    }


def wait(http, job):
    for _ in range(150):
        result = http.get(f"/api/jobs/{job['id']}").json()
        if result["status"] in {"succeeded", "failed", "cancelled"}:
            assert result["status"] == "succeeded", result
            return result["result"]
        time.sleep(0.02)
    pytest.fail("Video job timed out")


def test_tool_tracking_sample_import_preserves_source_and_existing_tracks(
    http, client, clip, tmp_path, monkeypatch
):
    project = client.post(
        "/api/projects",
        {
            "name": "CVAT sample",
            "labels": [{"id": 0, "name": "Background"}, {"id": 3, "name": "Snare"}],
        },
    )
    prefix = f"/api/projects/{project['id']}"
    catalog = client.get(prefix + "/dataset-templates")
    template = next(t for t in catalog if t["id"] == "hyperkvasir-tool-tracking")
    assert template["kind"] == "video" and template["category"] == "Video"
    assert not template["has_masks"] and template["importable"]
    assert "video" not in template and "url" not in template
    source = tmp_path / "sample.avi"
    source.write_bytes(clip)
    monkeypatch.setattr(
        http.app.state.services.dataset_templates.downloads, "fetch", lambda *args: source
    )
    request = {"template_id": template["id"]}
    result = client.wait(client.post(prefix + "/dataset-imports", request)["id"])
    assert result["asset_ids"] == [] and result["annotation_ids"] == []
    assert result["failed"] == [] and len(result["video_ids"]) == 1
    video = client.get(prefix + "/videos")[0]
    assert result["video_ids"] == [video["id"]]
    assert video["group_id"] == "hyperkvasir:99b387e7-d07b-4268-9226-4df450c2a198"
    assert (video["width"], video["height"], video["frames"]) == (64, 48, 6)
    assert client.get(prefix)["labels"] == project["labels"]
    path = f"/api/videos/{video['id']}"
    assert http.get(path + "/source").content == clip
    assert client.get(path + "/metadata")["timestamps"] == pytest.approx(
        [0, 0.1, 0.2, 0.5, 0.6, 0.7]
    )
    tracks = document()
    tracks["tracks"][0]["label_id"] = 3
    client.post(path + "/review", {"base_revision": 0, "document": tracks})
    client.post(path + "/decision", {"base_revision": 1, "verdict": "accepted"})
    saved = client.get(path + "/tracks")
    again = client.wait(client.post(prefix + "/dataset-imports", request)["id"])
    assert again["video_ids"] == result["video_ids"]
    assert len(client.get(prefix + "/videos")) == 1
    assert client.get(path + "/tracks") == saved
    assert len(client.get(prefix + "/decisions")) == 1
    assert client.get(prefix + "/assets") == []


@pytest.mark.parametrize(
    "choice",
    [
        {"include_masks": True},
        {"split": "train"},
        {"split": "validation"},
        {"evaluation_percentage": 20},
        {"offset": 1},
        {"channel": 1},
        {"targets": ["Snare"]},
        {"section": "test"},
    ],
)
def test_tool_tracking_sample_rejects_image_learning_choices_before_download(
    http, client, monkeypatch, choice
):
    project = client.post("/api/projects", {"name": "Video sample validation"})

    def forbidden(*args):
        raise AssertionError("Invalid video options must not start a download")

    monkeypatch.setattr(http.app.state.services.dataset_templates.downloads, "fetch", forbidden)
    response = http.post(
        f"/api/projects/{project['id']}/dataset-imports",
        json={"template_id": "hyperkvasir-tool-tracking", **choice},
    )
    assert response.status_code == 422
    assert "CVAT annotation" in response.json()["detail"]
    assert client.get(f"/api/projects/{project['id']}/jobs") == []


@pytest.fixture
def cvat(http):
    tasks = {}
    writes = []

    def respond(request):
        path = request.url.path
        if path == "/api/tasks" and request.method == "GET":
            return httpx.Response(200, json={"results": []})
        if path == "/api/tasks" and request.method == "POST":
            identifier = len(tasks) + 1
            tasks[identifier] = {"tags": [], "shapes": [], "tracks": []}
            return httpx.Response(201, json={"id": identifier})
        if path.startswith("/api/requests/"):
            return httpx.Response(200, json={"status": "finished"})
        if path == "/api/labels":
            return httpx.Response(
                200,
                json={"results": [{"name": "Grasper", "id": 11}, {"name": "Scissors", "id": 12}]},
            )
        if path == "/api/jobs":
            return httpx.Response(200, json={"results": [{"id": 101, "type": "annotation"}]})
        identifier = int(path.split("/")[3])
        if path.endswith("/data/"):
            return httpx.Response(202, json={"rq_id": f"import:task-{identifier}"})
        if path.endswith("/data/meta"):
            return httpx.Response(
                200,
                json={
                    "size": 6,
                    "start_frame": 0,
                    "stop_frame": 5,
                    "frames": [{"width": 64, "height": 48}],
                },
            )
        if path.endswith("/annotations"):
            if request.method == "PUT":
                writes.append(identifier)
                tasks[identifier] = json.loads(request.content)
                for i, track in enumerate(tasks[identifier]["tracks"]):
                    track["id"] = 900 + i
            return httpx.Response(200, json=tasks[identifier])
        raise AssertionError(path)

    adapter = CvatClient("http://cvat.test", "test-token")
    adapter.http.close()
    adapter.http = httpx.Client(base_url="http://cvat.test", transport=httpx.MockTransport(respond))
    http.app.state.services.video_editors.client = adapter
    http.app.state.services.video_editors.managed = False
    http.app.state.services.video_editors.public_url = "http://cvat.test"
    return tasks, writes


@pytest.mark.parametrize("failure", [500, 503, "connection"])
def test_cvat_waits_for_permissions_before_creating_a_task(monkeypatch, failure):
    calls = []

    def respond(request):
        calls.append(request.method)
        if len(calls) == 1:
            if failure == "connection":
                raise httpx.ConnectError("Service is starting", request=request)
            return httpx.Response(failure)
        return httpx.Response(200 if request.method == "GET" else 201, json={"id": 7})

    monkeypatch.setattr("monailabel.viewers.cvat.time.sleep", lambda _: None)
    client = CvatClient("http://cvat.test", "test-token")
    client.http.close()
    with httpx.Client(base_url=client.url, transport=httpx.MockTransport(respond)) as http:
        client.http = http
        assert client.create_task("Tool tracking", []) == 7
    assert calls == ["GET", "GET", "POST"]


@pytest.mark.parametrize("failure", [401, 403, "timeout", "write"])
def test_cvat_readiness_does_not_retry_denied_access_or_task_writes(monkeypatch, failure):
    calls = []

    def respond(request):
        calls.append(request.method)
        status = 500 if failure in {"timeout", "write"} else failure
        if failure == "write" and request.method == "GET":
            status = 200
        return httpx.Response(status, json={})

    clock = iter([0, 61])
    monkeypatch.setattr("monailabel.viewers.cvat.time.monotonic", lambda: next(clock))
    client = CvatClient("http://cvat.test", "test-token")
    client.http.close()
    with httpx.Client(base_url=client.url, transport=httpx.MockTransport(respond)) as http:
        client.http = http
        with pytest.raises(DomainError):
            client.create_task("Tool tracking", [])
    assert calls == (["GET", "POST"] if failure == "write" else ["GET"])


def test_video_import_timestamps_and_geometry_are_original(http, video, clip):
    prefix = f"/api/videos/{video['id']}"
    meta = http.get(prefix + "/metadata").json()
    assert meta["width"] == 64 and meta["height"] == 48
    assert meta["timestamps"] == pytest.approx([0, 0.1, 0.2, 0.5, 0.6, 0.7])
    assert video["frames"] == 6
    assert http.get(prefix + "/source").content == clip
    duplicate = http.post(
        f"/api/projects/{video['project_id']}/videos/upload",
        params={"name": "renamed.mp4", "group_id": "procedure-1"},
        content=clip,
    )
    assert duplicate.json()["id"] == video["id"]
    assert http.get(f"/api/projects/{video['project_id']}/assets").json() == []


def test_submitted_video_revisions_and_decisions_are_immutable(http, video):
    prefix = f"/api/videos/{video['id']}"
    request = {"base_revision": 0, "document": document()}
    first = http.post(prefix + "/review", json=request)
    assert first.status_code == 201
    assert http.post(prefix + "/review", json=request).status_code == 409
    assert (
        http.post(
            prefix + "/decision", json={"base_revision": 0, "verdict": "accepted"}
        ).status_code
        == 409
    )
    assert (
        http.post(
            prefix + "/decision", json={"base_revision": 1, "verdict": "accepted"}
        ).status_code
        == 200
    )
    second = http.post(
        prefix + "/review", json={"base_revision": 1, "document": {"tracks": []}}
    ).json()
    assert second["parent_id"] == first.json()["id"]
    assert len(http.get(prefix + "/revisions").json()) == 2
    assert (
        http.post(
            prefix + "/decision", json={"base_revision": 1, "verdict": "accepted"}
        ).status_code
        == 409
    )
    assert http.get(prefix + "/tracks").json()["document"]["tracks"] == []
    artifact = http.app.state.services.artifacts.read(first.json()["tracks_key"])
    assert TrackDocument.model_validate_json(artifact) == TrackDocument.model_validate(document())


@pytest.mark.parametrize("change", ["frame", "bounds", "label", "duplicate", "nan"])
def test_invalid_tracks_do_not_change_revision(http, video, change):
    doc = document()
    track = doc["tracks"][0]
    if change == "frame":
        track["keyframes"][-1]["frame"] = 6
    elif change == "bounds":
        track["keyframes"][0]["box"][2] = 65
    elif change == "label":
        track["label_id"] = 254
    elif change == "duplicate":
        doc["tracks"].append(copy.deepcopy(track))
    else:
        track["keyframes"][0]["box"][0] = "NaN"
    assert (
        http.post(
            f"/api/videos/{video['id']}/review", json={"base_revision": 0, "document": doc}
        ).status_code
        == 422
    )
    assert http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0


def test_cvat_drafts_survive_reopen_and_stale_submission(http, video, cvat):
    tasks, writes = cvat
    prefix = f"/api/videos/{video['id']}"
    job = http.post(prefix + "/editor", json={"base_revision": 0}).json()
    opened = wait(http, job)
    assert opened["url"] == "http://cvat.test/tasks/1/jobs/101"
    payload = encode_tracks(TrackDocument.model_validate(document()), {1: 11})
    payload["tracks"][0]["id"] = 700
    tasks[1] = payload
    reopened = wait(http, http.post(prefix + "/editor", json={"base_revision": 0}).json())
    assert reopened == opened
    assert writes == [1]
    submitted = http.post(prefix + "/cvat-submit", json={"editor_id": opened["editor_id"]})
    assert submitted.status_code == 201, submitted.text
    assert (
        http.post(prefix + "/cvat-submit", json={"editor_id": opened["editor_id"]}).json()
        == submitted.json()
    )
    saved = http.get(prefix + "/tracks").json()["document"]
    assert saved["tracks"][0]["id"] == opened["editor_id"] + "-700"
    review = wait(
        http, http.post(prefix + "/editor", json={"base_revision": 1, "mode": "review"}).json()
    )
    edit = wait(http, http.post(prefix + "/editor", json={"base_revision": 1}).json())
    assert edit["editor_id"] != review["editor_id"]
    assert tasks[2]["tracks"][0]["shapes"] == payload["tracks"][0]["shapes"]
    assert (
        http.post(prefix + "/cvat-submit", json={"editor_id": edit["editor_id"]}).status_code == 201
    )
    assert http.get(prefix + "/tracks").json()["document"] == saved
    before = copy.deepcopy(tasks[2])
    assert (
        http.post(prefix + "/cvat-submit", json={"editor_id": review["editor_id"]}).status_code
        == 409
    )
    assert tasks[2] == before


def test_video_discovery_continues_to_the_requested_editor(http, video, cvat):
    from monailabel.core.chat import ChatMessage, ToolCall

    planner = http.app.state.services.assistants.provider
    planner.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=f"discover-{index}",
                    name=name,
                    arguments=args,
                )
            ],
        )
        for index, (name, args) in enumerate(
            [
                ("list_videos", {}),
                ("open_video_editor", {"video_id": video["id"], "base_revision": 0}),
            ]
        )
    ]
    response = http.post(
        f"/api/projects/{video['project_id']}/assistant", json={"message": "Open the video in CVAT"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["tools"] == ["list_videos", "open_video_editor"]
    assert wait(http, {"id": response.json()["job_id"]})["video_id"] == video["id"]


def test_queued_editor_launch_rejects_a_new_revision_before_resuming_draft(http, video, cvat):
    from monailabel.core.models import Project
    from monailabel.core.video import VideoEditorRequest

    service = http.app.state.services
    prefix = f"/api/videos/{video['id']}"
    wait(http, http.post(prefix + "/editor", json={"base_revision": 0}).json())
    before = copy.deepcopy(cvat[0])
    http.post(
        prefix + "/review", json={"base_revision": 0, "document": document()}
    ).raise_for_status()
    # The launch was queued with revision 0, but another editor submitted before it ran.
    with pytest.raises(DomainError, match="submitted video changed"):
        service.video_editors.prepare(
            VideoAsset.model_validate(video),
            service.store.get(Project, video["project_id"]),
            VideoEditorRequest(base_revision=0),
            None,
        )
    assert cvat[0] == before
    assert http.get(prefix + "/tracks").json()["base_revision"] == 1


@pytest.mark.parametrize(
    "unsupported", ["rotation", "invalid_polygon", "tag", "group", "attribute"]
)
def test_cvat_rejects_unsupported_edits_without_silent_loss(unsupported):
    payload = encode_tracks(TrackDocument.model_validate(document()), {1: 11})
    track = payload["tracks"][0]
    track["id"] = 50
    if unsupported == "rotation":
        track["shapes"][0]["rotation"] = 15
    elif unsupported == "invalid_polygon":
        track["shapes"][0]["type"] = "polygon"
    elif unsupported == "tag":
        payload["tags"] = [{"label_id": 1}]
    elif unsupported == "group":
        track["group"] = 1
    else:
        track["attributes"] = [{"spec_id": 1, "value": "important"}]
    with pytest.raises(DomainError, match="draft has been kept"):
        decode_tracks(payload, {11: 1}, {}, "task")


def image_bytes():
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def test_procedure_splits_work_in_both_import_directions(http, video, clip):
    pid = video["project_id"]
    response = http.post(
        f"/api/projects/{pid}/assets/upload",
        params={"name": "frame.png", "group_id": "procedure-1", "split": "train"},
        content=image_bytes(),
    )
    assert response.status_code == 409
    response = http.post(
        f"/api/projects/{pid}/assets/upload",
        params={"name": "frame.png", "group_id": "procedure-1"},
        content=image_bytes(),
    )
    assert response.status_code == 201
    image = response.json()
    assert http.post(f"/api/assets/{image['id']}/assign-validation").status_code == 200
    assert http.get(f"/api/projects/{pid}/videos").json()[0]["split"] == "validation"
    assert (
        http.post(
            f"/api/projects/{pid}/videos/upload",
            params={"name": "clip.mp4", "group_id": "procedure-1", "split": "train"},
            content=clip,
        ).status_code
        == 409
    )


def test_reserved_video_procedure_cannot_enter_image_training(http, client, clip):
    project = client.post("/api/projects", {"name": "Held-out video"})
    pid = project["id"]
    response = http.post(
        f"/api/projects/{pid}/videos/upload",
        params={"name": "clip.mp4", "group_id": "held-out", "split": "validation"},
        content=clip,
    )
    assert response.status_code == 201
    assert (
        http.post(
            f"/api/projects/{pid}/assets/upload",
            params={"name": "frame.png", "group_id": "held-out", "split": "train"},
            content=image_bytes(),
        ).status_code
        == 409
    )
    assert (
        http.post(
            f"/api/projects/{pid}/assets/upload",
            params={"name": "frame.png", "group_id": "held-out"},
            content=image_bytes(),
        ).json()["split"]
        == "validation"
    )


def test_upload_limits_and_invalid_video(http, video, monkeypatch):
    path = f"/api/projects/{video['project_id']}/videos/upload"
    params = {"name": "bad.mp4", "group_id": "other"}
    monkeypatch.setattr("monailabel.server.video_api.MAX_VIDEO_BYTES", 32)
    for content in (b"x" * 33, iter([b"x" * 20, b"x" * 20])):
        assert http.post(path, params=params, content=content).status_code == 413
    for content in (b"", b"invalid"):
        assert http.post(path, params=params, content=content).status_code == 422
    assert len(http.get(f"/api/projects/{video['project_id']}/videos").json()) == 1


@pytest.mark.parametrize("role", ["annotator", "reviewer", "unrelated"])
def test_video_access_controls(http, client, video, role):
    pid, vid = video["project_id"], video["id"]
    user = client.post("/api/auth/users", {"username": role, "password": "test-password-1234"})
    if role != "unrelated":
        client.request(
            "PUT", f"/api/projects/{pid}/members", {"user_id": user["id"], "roles": [role]}
        )
    http.post("/api/auth/login", json={"username": role, "password": "test-password-1234"})
    assert http.get(f"/api/videos/{vid}/tracks").status_code == (
        403 if role == "unrelated" else 200
    )
    assert http.get(f"/api/videos/{vid}/source").status_code == (
        403 if role == "unrelated" else 200
    )
    assert (
        http.post(
            f"/api/projects/{pid}/videos/upload",
            params={"name": "x", "group_id": "p"},
            content=b"x",
        ).status_code
        == 403
    )
    assert http.post(
        f"/api/videos/{vid}/review", json={"base_revision": 0, "document": document()}
    ).status_code == (201 if role == "annotator" else 403)
    assert http.post(
        f"/api/videos/{vid}/decision", json={"base_revision": 0, "verdict": "accepted"}
    ).status_code == (409 if role == "reviewer" else 403)
    http.app.state.services.video_editors.managed = False
    assert http.post(f"/api/videos/{vid}/editor", json={"base_revision": 0}).status_code == (
        503 if role == "annotator" else 403
    )


def test_video_project_deletion_removes_tracks_and_blobs(http, video):
    store = http.app.state.services.store
    assert len(store.list(VideoAsset, video["project_id"])) == 1
    assert (
        http.request(
            "DELETE",
            f"/api/projects/{video['project_id']}",
            json={"confirmation_name": "Video regression"},
        ).status_code
        == 200
    )
    assert store.list(VideoAsset, video["project_id"]) == []


def test_clip_deletion_preserves_other_clips_and_external_cvat_drafts(http, video, cvat):
    from monailabel.server.deletion import cleanup_storage

    tasks, _ = cvat
    service = http.app.state.services
    opened = wait(
        http, http.post(f"/api/videos/{video['id']}/editor", json={"base_revision": 0}).json()
    )
    before = copy.deepcopy(tasks)
    response = http.request(
        "DELETE", f"/api/projects/{video['project_id']}/assets", json={"asset_ids": [video["id"]]}
    )
    assert response.status_code == 200
    assert service.store.list(VideoAsset, video["project_id"]) == []
    assert tasks == before
    assert (
        http.post(
            f"/api/videos/{video['id']}/cvat-submit", json={"editor_id": opened["editor_id"]}
        ).status_code
        == 404
    )
    cleanup_storage(service.store, service.artifacts)
    assert not service.artifacts.path(video["source_key"]).exists()
    with pytest.raises(DomainError), service.store.transaction() as session:
        session.insert(VideoAsset.model_validate(video))


def test_cvat_frame_deletion_or_resampling_is_rejected(video):
    asset = VideoAsset.model_validate(video)
    client = CvatClient("http://cvat.test", "test-token")
    client.http.close()
    meta = {"size": 6, "start_frame": 0, "stop_frame": 5, "frames": [{"width": 64, "height": 48}]}
    for changed in (
        {"deleted_frames": [2]},
        {"frame_filter": "step=2"},
        {"size": 5},
        {"frames": [{"width": 32, "height": 48}]},
        {"included_frames": [0, 2, 4]},
    ):
        with httpx.Client(
            base_url="http://cvat.test",
            transport=httpx.MockTransport(
                lambda _, changes=changed: httpx.Response(200, json=meta | changes)
            ),
        ) as transport:
            client.http = transport
            with pytest.raises(DomainError, match="source frame grid"):
                client.check_frames(1, asset)


def test_original_timestamp_origin_is_retained(monkeypatch, tmp_path):
    from monailabel.server.videos import probe_video

    data = {
        "streams": [{"width": 64, "height": 48, "codec_name": "h264"}],
        "frames": [
            {"best_effort_timestamp_time": t, "duration_time": "0.1"}
            for t in ("12.0", "12.1", "12.5")
        ],
    }
    monkeypatch.setattr(
        "monailabel.server.videos.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout=json.dumps(data)),
    )
    metadata = probe_video(tmp_path / "clip.mp4")
    assert metadata.start_time == 12
    assert metadata.timestamps == pytest.approx([0, 0.1, 0.5])


def test_video_tools_check_project_access_and_return_real_jobs(http, client, video, cvat):
    from monailabel.core.chat import ToolCall
    from monailabel.core.models import AssistantContext, User
    from monailabel.server.assistant_tools import catalog
    from monailabel.server.assistant_tools.base import ToolContext

    service = http.app.state.services
    user = service.store.list(User)[0]
    registry = catalog(
        ToolContext(service, video["project_id"], user, AssistantContext(), "Open video")
    )
    listed = registry.execute(ToolCall(id="list", name="list_videos", arguments={}))
    assert listed.data["videos"][0]["id"] == video["id"]
    opened = registry.execute(
        ToolCall(
            id="open",
            name="open_video_editor",
            arguments={"video_id": video["id"], "base_revision": 0},
        )
    )
    result = wait(http, {"id": opened.job_id})
    assert result["video_id"] == video["id"]
    other = client.post("/api/projects", {"name": "Other video project"})
    foreign = catalog(ToolContext(service, other["id"], user, AssistantContext(), "Open video"))
    with pytest.raises(DomainError, match="current project"):
        foreign.execute(
            ToolCall(
                id="wrong-project",
                name="open_video_editor",
                arguments={"video_id": video["id"], "base_revision": 0},
            )
        )


def test_deleting_unused_evaluation_clip_releases_its_procedure(http, client, clip):
    pid = client.post("/api/projects", {"name": "Disposable evaluation clip"})["id"]
    params = {"name": "clip.mp4", "group_id": "procedure", "split": "validation"}
    video = http.post(f"/api/projects/{pid}/videos/upload", params=params, content=clip).json()
    removed = http.request(
        "DELETE", f"/api/projects/{pid}/assets", json={"asset_ids": [video["id"]]}
    )
    assert removed.status_code == 200
    assert (
        http.post(
            f"/api/projects/{pid}/videos/upload", params=params | {"split": "pool"}, content=clip
        ).status_code
        == 201
    )


@pytest.fixture
def managed_editor(http, video):
    from monailabel.server.video_editor import VideoEditor

    service = http.app.state.services
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={"tracks": [], "shapes": [], "tags": []})

    client = CvatClient("http://private-cvat.test", "private-service-token")
    client.http.close()
    client.http = httpx.Client(
        base_url=client.url,
        headers={"Authorization": "Token private-service-token"},
        transport=httpx.MockTransport(upstream),
    )
    service.video_editors.client = client
    editor = VideoEditor(
        project_id=video["project_id"],
        asset_id=video["id"],
        base_revision=0,
        mode="annotation",
        server_url=client.url,
        task_id=11,
        job_id=21,
        ready=True,
        label_map={1: 101, 2: 102},
    )
    with service.store.transaction() as session:
        session.insert(editor)
    return editor, calls


def test_managed_cvat_proxy_scopes_every_task_and_keeps_service_auth_private(
    http, client, video, managed_editor
):
    editor, calls = managed_editor
    pid = video["project_id"]
    user = client.post(
        "/api/auth/users", {"username": "annotator", "password": "test-password-1234"}
    )
    client.request(
        "PUT", f"/api/projects/{pid}/members", {"user_id": user["id"], "roles": ["annotator"]}
    )
    http.post("/api/auth/login", json={"username": "annotator", "password": "test-password-1234"})
    assert http.get(f"/api/cvat/editors/{editor.id}").status_code == 200
    response = http.get("/cvat-api/jobs/21/annotations")
    assert response.status_code == 200
    assert calls[-1].headers["Authorization"] == "Token private-service-token"
    assert "private-service-token" not in response.text
    assert "set-cookie" not in response.headers
    for path in ("/cvat-api/tasks/999", "/cvat-api/jobs/999/data?type=frame&number=0"):
        assert http.get(path).status_code == 404
    for path in (
        "/cvat-api/tasks?job_id=21",
        "/cvat-api/labels?task_id=11&job_id=999",
        "/cvat-api/jobs/21/annotations?task_id=999",
        "/cvat-api/labels?task_id=11&filter=anything",
    ):
        assert http.get(path).status_code == 400
    assert http.delete("/cvat-api/tasks/11").status_code == 403
    assert http.patch("/cvat-api/users/1", json={"is_superuser": True}).status_code == 403
    assert len(calls) == 1
    http.post("/api/auth/logout")
    assert http.get("/cvat-api/jobs/21/annotations").status_code == 401


def test_tracking_proposal_does_not_change_draft_and_rejects_stale_revision(
    http, video, managed_editor
):
    from monailabel.core.video import TrackKeyframe

    class Tracker:
        def track(
            self, source, width, height, seed, frame_count, progress, output="box", seed_mask=None
        ):
            assert source.is_file() and (width, height) == (64, 48)
            return VideoTrackingResult(
                [TrackKeyframe(frame=i, box=[i + 1, 2, i + 10, 20]) for i in range(frame_count)],
                {},
                [],
            )

    editor, calls = managed_editor
    service = http.app.state.services
    service.video_tracking.provider = Tracker()
    request = {
        "editor_id": editor.id,
        "base_revision": 0,
        "client_id": 3,
        "label_id": 1,
        "seed": {"frame": 0, "box": [1, 2, 10, 20]},
        "frame_count": 3,
        "draft_signature": "a" * 64,
    }
    response = http.post(f"/api/videos/{video['id']}/track", json=request)
    assert response.status_code == 202
    result = wait(http, response.json())
    path = f"/api/videos/{video['id']}/tracking-proposals/{result['video_proposal_id']}"
    proposal = http.get(path).json()
    assert proposal["request"] == {
        **request,
        "output": "box",
        "seed": {**request["seed"], "outside": False, "occluded": False},
    }
    assert [key["frame"] for key in proposal["keyframes"]] == [0, 1, 2]
    assert proposal["provider"] == "sam2" and len(proposal["model_checksum"]) == 64
    assert not calls
    assert http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0
    assert (
        http.post(
            f"/api/videos/{video['id']}/track", json={**request, "frame_count": 65}
        ).status_code
        == 422
    )
    assert (
        http.post(
            f"/api/videos/{video['id']}/track", json={**request, "frame_count": 7}
        ).status_code
        == 422
    )
    assert (
        http.post(
            f"/api/videos/{video['id']}/review", json={"base_revision": 0, "document": document()}
        ).status_code
        == 201
    )
    assert http.get(path).status_code == 409
    assert http.post(f"/api/videos/{video['id']}/track", json=request).status_code == 409


def tracking_chat(http, project_id, context=None):
    from monailabel.core.chat import ChatMessage, ToolCall

    http.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="track", name="track_selected_video_tool", arguments={"frame_count": 3})
            ],
        )
    )
    response = http.post(
        f"/api/projects/{project_id}/assistant",
        json={"message": "Track tool for 3 frames", "context": context or {}},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_workspace_tracking_chat_opens_the_only_clip_without_running_inference(http, video, cvat):
    tasks, writes = cvat
    first = tracking_chat(http, video["project_id"])
    opened = wait(http, {"id": first["job_id"]})
    assert opened["video_id"] == video["id"]
    assert "choose Track" in first["message"]
    assert "Tracking has not started" in first["message"]
    before = copy.deepcopy(tasks)
    again = tracking_chat(http, video["project_id"])
    resumed = wait(http, {"id": again["job_id"]})
    assert resumed["editor_id"] == opened["editor_id"]
    assert tasks == before and writes == [1]
    jobs = http.get(f"/api/projects/{video['project_id']}/jobs").json()
    assert {job["kind"] for job in jobs} == {"video_editor"}


def test_cvat_chat_without_a_selected_track_returns_guidance_and_keeps_the_draft(
    http, video, managed_editor
):
    editor, calls = managed_editor
    context = {
        "base_revision": 0,
        "video": {
            "video_id": video["id"],
            "editor_id": editor.id,
            "frame": 0,
            "draft_signature": "a" * 64,
        },
    }
    reply = tracking_chat(http, video["project_id"], context)
    assert reply["job_id"] is None
    assert "choose Track" in reply["message"]
    assert "Track this tool for 3 frames" in reply["message"]
    assert not calls
    assert http.get(f"/api/projects/{video['project_id']}/jobs").json() == []


@pytest.mark.parametrize("count", [0, 2])
def test_workspace_tracking_chat_does_not_guess_a_clip(http, client, video, monkeypatch, count):
    service = http.app.state.services
    original = service.store.list
    sample = VideoAsset.model_validate(video)

    def listing(model, *args):
        if model is VideoAsset:
            return [sample.model_copy(update={"id": f"clip-{i}"}) for i in range(count)]
        return original(model, *args)

    monkeypatch.setattr(service.store, "list", listing)
    reply = tracking_chat(http, video["project_id"])
    assert reply["job_id"] is None
    assert ("Choose the clip" if count else "Import a video") in reply["message"]
    assert client.get(f"/api/projects/{video['project_id']}/jobs") == []


def test_managed_cvat_denies_other_projects_and_review_writes(http, client, video, managed_editor):

    editor, calls = managed_editor
    review = editor.model_copy(
        update={"id": "review-task", "mode": "review", "task_id": 12, "job_id": 22}
    )
    with http.app.state.services.store.transaction() as session:
        session.insert(review)
    unrelated = client.post(
        "/api/auth/users", {"username": "outsider", "password": "test-password-1234"}
    )
    annotator = client.post(
        "/api/auth/users", {"username": "writer", "password": "test-password-1234"}
    )
    client.request(
        "PUT",
        f"/api/projects/{video['project_id']}/members",
        {"user_id": annotator["id"], "roles": ["annotator"]},
    )
    http.post(
        "/api/auth/login",
        json={"username": unrelated["username"], "password": "test-password-1234"},
    )
    assert http.get("/cvat-api/jobs/21/annotations").status_code == 403
    assert http.get("/cvat-api/labels?task_id=11").status_code == 403
    assert http.get(f"/api/cvat/editors/{editor.id}").status_code == 403
    http.post("/api/auth/login", json={"username": "writer", "password": "test-password-1234"})
    assert (
        http.patch("/cvat-api/jobs/22/annotations?action=update", json={"tracks": []}).status_code
        == 403
    )
    assert (
        http.post(
            f"/api/videos/{video['id']}/cvat-submit", json={"editor_id": review.id}
        ).status_code
        == 403
    )
    assert not calls


@pytest.fixture
def find_tracking(http, client, video, managed_editor):
    from monailabel.core.video import ToolDetection, TrackKeyframe

    service = http.app.state.services
    model = client.post(
        f"/api/projects/{video['project_id']}/models",
        {
            "name": "GPT-5.6 Sol",
            "provider": "openai-chat-polygons",
            "config": {"url": "https://unused.test", "model": "sol-fixture"},
        },
    )
    calls = []

    class Detector:
        def locate(self, image, label, prompt, chosen):
            assert image.shape == (48, 64, 3) and image.dtype == np.float32
            assert label.id == 1 and label.name == "Grasper"
            calls.append((chosen.id, prompt))
            return ToolDetection(status="found", box=[1.5, 2, 20, 30])

    class Tracker:
        def track(
            self, source, width, height, seed, frame_count, progress, output="box", seed_mask=None
        ):
            calls.append((seed.frame, frame_count))
            return VideoTrackingResult(
                [TrackKeyframe(frame=seed.frame + i, box=seed.box) for i in range(frame_count)],
                {},
                [],
            )

    service.video_tracking.detector = Detector()
    service.video_tracking.provider = Tracker()
    request = {
        "editor_id": managed_editor[0].id,
        "base_revision": 0,
        "model_id": model["id"],
        "label_id": 1,
        "frame": 2,
        "frame_count": 3,
        "prompt": "leftmost grasper",
        "draft_signature": "b" * 64,
    }
    return request, calls


def test_find_tracking_without_rectangle_and_with_model_provenance(
    http, video, managed_editor, find_tracking
):
    request, calls = find_tracking
    response = http.post(f"/api/videos/{video['id']}/find-and-track", json=request)
    assert response.status_code == 202, response.text
    result = wait(http, response.json())
    path = f"/api/videos/{video['id']}/tracking-proposals/{result['video_proposal_id']}"
    proposal = http.get(path).json()
    assert proposal["request"]["client_id"] is None
    assert proposal["request"]["seed"]["box"] == [1.5, 2, 20, 30]
    assert proposal["request"]["draft_signature"] == request["draft_signature"]
    assert [key["frame"] for key in proposal["keyframes"]] == [2, 3, 4]
    assert proposal["detection"]["model_id"] == request["model_id"]
    assert proposal["detection"]["remote_model"] == "sol-fixture"
    assert calls == [(request["model_id"], "leftmost grasper"), (2, 3)]
    assert not managed_editor[1]
    assert http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0
    http.post(
        f"/api/videos/{video['id']}/review", json={"base_revision": 0, "document": document()}
    ).raise_for_status()
    assert http.get(path).status_code == 409
    assert http.post(f"/api/videos/{video['id']}/find-and-track", json=request).status_code == 409
    assert len(calls) == 2


@pytest.mark.parametrize(
    "status,box", [("not_found", None), ("ambiguous", None), ("found", [1, 2, 100, 30])]
)
def test_detection_failure_never_tracks_or_changes_drafts(
    http, video, managed_editor, find_tracking, status, box
):
    from monailabel.core.video import ToolDetection, VideoTrackingProposal

    request, calls = find_tracking
    service = http.app.state.services

    class Detector:
        def locate(self, *args):
            return ToolDetection(status=status, box=box)

    service.video_tracking.detector = Detector()
    job = http.post(f"/api/videos/{video['id']}/find-and-track", json=request).json()
    for _ in range(150):
        result = http.get(f"/api/jobs/{job['id']}").json()
        if result["status"] == "failed":
            break
        time.sleep(0.02)
    assert result["status"] == "failed", result
    assert not calls and not managed_editor[1]
    assert service.store.list(VideoTrackingProposal, video["project_id"]) == []


def test_find_tracking_validates_range_label_model_and_editor_before_inference(
    http, client, video, find_tracking
):
    from monailabel.core.models import ModelRecord

    request, calls = find_tracking
    service = http.app.state.services
    other = client.post("/api/projects", {"name": "Other detector project"})
    foreign = ModelRecord(
        project_id=other["id"], name="Foreign", provider="openai-polygons", label_ids=[0]
    )
    local = ModelRecord(
        project_id=video["project_id"], name="Local", provider="sam2", label_ids=[0]
    )
    with service.store.transaction() as session:
        session.insert(foreign)
        session.insert(local)
    for changes, status in [
        ({"frame": 5}, 422),
        ({"label_id": 3}, 422),
        ({"model_id": foreign.id}, 403),
        ({"model_id": local.id}, 422),
    ]:
        response = http.post(
            f"/api/videos/{video['id']}/find-and-track", json={**request, **changes}
        )
        assert response.status_code == status, response.text
    assert not calls


def test_find_tracking_chat_honors_named_model_and_selected_label(
    http, client, video, find_tracking
):
    from monailabel.core.chat import ChatMessage, ToolCall

    request, calls = find_tracking
    astra = client.post(
        f"/api/projects/{video['project_id']}/models",
        {
            "name": "GPT-6 Astra",
            "provider": "openai-polygons",
            "config": {"url": "https://unused.test", "model": "astra-fixture"},
        },
    )
    http.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="find-tool",
                    name="find_and_track_video_tool",
                    arguments={
                        "model_name": "GPT-6 Astra",
                        "frame_count": 3,
                        "prompt": "leftmost grasper",
                    },
                ),
            ],
        )
    )
    response = http.post(
        f"/api/projects/{video['project_id']}/assistant",
        json={
            "message": "Use GPT-6 Astra to find the grasper and track it for 3 frames",
            "context": {
                "base_revision": 0,
                "model_id": request["model_id"],
                "label_ids": [1],
                "video": {
                    "video_id": video["id"],
                    "editor_id": request["editor_id"],
                    "frame": 2,
                    "draft_signature": request["draft_signature"],
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    wait(http, {"id": response.json()["job_id"]})
    assert calls == [(astra["id"], "leftmost grasper"), (2, 3)]


@pytest.mark.parametrize("invalid", ["id_as_name", "alias", "conflicting_id"])
def test_video_chat_repairs_model_selector_before_starting_one_job(
    http, client, video, find_tracking, invalid
):
    from monailabel.core.chat import ChatMessage, ToolCall

    request, calls = find_tracking
    planner = http.app.state.services.assistants.provider
    selector = {"model_name": request["model_id"] if invalid == "id_as_name" else "GPT Sol"}
    if invalid == "conflicting_id":
        selector = {"model_name": "GPT-5.6 Sol", "model_id": "wrong-id"}
    planner.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=f"repair-{index}",
                    name="find_and_track_video_tool",
                    arguments={"frame_count": 3, **args},
                )
            ],
        )
        for index, args in enumerate([selector, {}])
    ]
    response = http.post(
        f"/api/projects/{video['project_id']}/assistant",
        json={
            "message": "Locate the grasper and track for 3 frames",
            "context": {
                "base_revision": 0,
                "model_id": request["model_id"],
                "label_ids": [1],
                "video": {
                    "video_id": video["id"],
                    "editor_id": request["editor_id"],
                    "frame": 2,
                    "draft_signature": request["draft_signature"],
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    wait(http, {"id": response.json()["job_id"]})
    assert calls == [(request["model_id"], ""), (2, 3)]
    assert response.json()["tools"] == ["find_and_track_video_tool"]
    assert len(planner.calls) == 2
    assert "omit model_name and model_id" in planner.calls[-1][0][-1].content
    assert http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0


@pytest.mark.parametrize("clarify", [True, False])
def test_unavailable_video_model_never_starts_annotation(http, video, find_tracking, clarify):
    from monailabel.core.chat import ChatMessage, ToolCall

    request, calls = find_tracking
    planner = http.app.state.services.assistants.provider
    missing = ChatMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="missing-model",
                name="find_and_track_video_tool",
                arguments={
                    "model_name": "Unavailable model",
                    "output": "polygon",
                    "frame_count": 3,
                },
            )
        ],
    )
    planner.queue = [missing]
    if clarify:
        planner.queue.append(
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="clarify-model",
                        name="clarify_request",
                        arguments={
                            "question": "That model is unavailable. Which model should I use?"
                        },
                    )
                ],
            )
        )
    else:
        planner.queue.extend([missing, missing])
    response = http.post(
        f"/api/projects/{video['project_id']}/assistant",
        json={
            "message": "Use Unavailable model to segment the tool and track for 3 frames",
            "context": {
                "base_revision": 0,
                "model_id": request["model_id"],
                "label_ids": [1],
                "video": {
                    "video_id": video["id"],
                    "editor_id": request["editor_id"],
                    "frame": 2,
                    "draft_signature": request["draft_signature"],
                },
            },
        },
    )
    assert response.status_code == (200 if clarify else 422), response.text
    if clarify:
        assert response.json()["tools"] == ["clarify_request"]
        assert response.json()["job_id"] is None
    else:
        assert len(planner.calls) == 3  # Bounded retries; no substitute model or action.
    assert not calls
    jobs = http.get(f"/api/projects/{video['project_id']}/jobs").json()
    assert not any(job["kind"] == "video_find_tracking" for job in jobs)


@pytest.mark.parametrize(
    "scope,expected_frame,expected_count",
    [("current_frame", 2, 1), ("frames", 2, 3), ("whole_video", 0, 6)],
)
def test_video_prompt_frame_scopes(
    http, video, find_tracking, scope, expected_frame, expected_count
):
    from monailabel.core.chat import ChatMessage, ToolCall

    request, calls = find_tracking
    http.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="scope",
                    name="find_and_track_video_tool",
                    arguments={"scope": scope, "frame_count": 3},
                ),
            ],
        )
    )
    response = http.post(
        f"/api/projects/{video['project_id']}/assistant",
        json={
            "message": "Locate the grasper",
            "context": {
                "base_revision": 0,
                "model_id": request["model_id"],
                "label_ids": [1],
                "video": {
                    "video_id": video["id"],
                    "editor_id": request["editor_id"],
                    "frame": 2,
                    "draft_signature": "a" * 64,
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    result = wait(http, {"id": response.json()["job_id"]})
    proposal = http.get(
        f"/api/videos/{video['id']}/tracking-proposals/{result['video_proposal_id']}"
    ).json()
    assert proposal["request"]["seed"]["frame"] == expected_frame
    assert len(proposal["keyframes"]) == expected_count
    if expected_count == 1:
        assert len(calls) == 1  # The selected annotation model runs; the tracker does not.
        assert proposal["provider"] == "openai-chat-polygons"
        assert proposal["model_checksum"] is None
    else:
        assert calls[-1] == (expected_frame, expected_count)


@pytest.mark.parametrize("provider", ["openai-chat-polygons", "http-mask", "huggingface"])
@pytest.mark.parametrize("output", ["box", "polygon"])
def test_single_frame_segmentation_uses_chosen_model_and_preserves_masks(
    http, video, find_tracking, provider, output
):
    import zipfile

    from monailabel.core.ports import Prediction

    request, calls = find_tracking
    service = http.app.state.services
    model = http.post(
        f"/api/projects/{video['project_id']}/models",
        json={
            "name": "Chosen annotation model",
            "provider": provider,
            "label_ids": [0, 1],
            "config": {"url": "https://unused.test"},
        },
    )
    assert model.status_code == 201, model.text
    request = {**request, "model_id": model.json()["id"]}
    mask = np.zeros((48, 64), np.uint8)
    mask[4:30, 8:40] = 1
    mask[10:20, 15:25] = 0  # A hole must be disclosed, and the exact mask retained.

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            assert model.id == request["model_id"] and model.label_ids == [0, 1]
            assert "Grasper" in prompt
            return Prediction(mask)

    service.models.providers[provider] = Segmenter()
    response = http.post(
        f"/api/videos/{video['id']}/find-and-track",
        json={**request, "frame_count": 1, "output": output},
    )
    result = wait(http, response.json())
    path = f"/api/videos/{video['id']}/tracking-proposals/{result['video_proposal_id']}"
    proposal = http.get(path).json()
    assert proposal["request"]["output"] == output
    assert proposal["provider"] == provider
    if output == "polygon":
        assert "points" in proposal["keyframes"][0] and "box" not in proposal["keyframes"][0]
        assert proposal["warnings"] and proposal["masks_key"]
        assert calls == []  # Neither the box detector nor SAM is invoked.
        with zipfile.ZipFile(io.BytesIO(http.get(path + "/masks").content)) as bundle:
            original = np.asarray(Image.open(io.BytesIO(bundle.read("000002.png"))))
            assert np.array_equal(original, mask > 0)
    else:
        expected = [1.5, 2, 20, 30] if provider == "openai-chat-polygons" else [8, 4, 40, 30]
        assert proposal["keyframes"][0]["box"] == expected
        assert len(calls) == (1 if provider == "openai-chat-polygons" else 0)
    assert http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0


def test_polygon_tracks_roundtrip_and_revision_geometry(http, video, cvat):
    from monailabel.core.video import PolygonKeyframe

    key = PolygonKeyframe(frame=1, points=[1.25, 2, 20.5, 3.5, 10.75, 30])
    doc = document()
    doc["tracks"].append(
        {
            "id": "polygon-tool",
            "label_id": 2,
            "keyframes": [
                key.model_dump(),
                key.model_copy(update={"frame": 4, "outside": True}).model_dump(),
            ],
        }
    )
    submitted = http.post(
        f"/api/videos/{video['id']}/review", json={"base_revision": 0, "document": doc}
    )
    assert submitted.status_code == 201, submitted.text
    opened = wait(
        http,
        http.post(
            f"/api/videos/{video['id']}/editor", json={"base_revision": 1, "mode": "review"}
        ).json(),
    )
    assert cvat[0][1]["tracks"][1]["shapes"][0]["type"] == "polygon"
    assert cvat[0][1]["tracks"][1]["shapes"][0]["points"] == key.points
    response = http.post(
        f"/api/videos/{video['id']}/cvat-submit", json={"editor_id": opened["editor_id"]}
    )
    assert response.status_code == 201, response.text
    saved = http.get(f"/api/videos/{video['id']}/tracks").json()["document"]
    assert saved == TrackDocument.model_validate(doc).model_dump()
