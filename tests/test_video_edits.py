"""CVAT chat edits are authorized plans, never silent changes to saved tracks."""

import pytest
import test_videos

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.server.video.models import VideoEditor

clip = test_videos.clip
isolated_cvat = test_videos.isolated_cvat
managed_editor = test_videos.managed_editor
video = test_videos.video


def edit(http, video, editor, arguments, *, context=None, tool="clear_video_annotations"):
    http.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="edit", name=tool, arguments=arguments)],
        )
    )
    return http.post(
        f"/api/projects/{video['project_id']}/assistant",
        json={
            "message": "Clear the requested frames"
            if tool == "clear_video_annotations"
            else "Undo",
            "context": {
                "base_revision": 0,
                "video": {
                    "video_id": video["id"],
                    "editor_id": editor.id,
                    "frame": 2,
                    "draft_signature": "a" * 64,
                    "client_id": 10,
                    "label_id": 1,
                },
                "viewer_actions": ["clear_video_annotations", "undo", "redo", "submit"],
                **(context or {}),
            },
        },
    )


@pytest.mark.parametrize(
    "scope,count,expected",
    [
        ("current_frame", 1, (2, 3)),
        ("frames", 3, (2, 5)),
        ("whole_video", 1, (0, 6)),
    ],
)
def test_clearing_plans_preserve_requested_scope_without_saving(
    http, client, video, managed_editor, scope, count, expected
):
    editor, calls = managed_editor
    response = edit(
        http, video, editor, {"targets": ["grasper"], "scope": scope, "frame_count": count}
    )
    assert response.status_code == 200, response.text
    reply = response.json()
    action = reply["data"]
    assert reply["tools"] == ["clear_video_annotations"] and reply["job_id"] is None
    assert (action["start"], action["stop"]) == expected
    assert action["label_ids"] == [1] and action["client_id"] is None
    assert action["draft_signature"] == "a" * 64
    assert action["base_revision"] == 0 and action["editor_id"] == editor.id
    assert client.get(f"/api/videos/{video['id']}/tracks")["base_revision"] == 0
    assert not calls


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"targets": ["missing"]},
        {"all_targets": True, "targets": ["Grasper"]},
        {"all_targets": True, "scope": "frames", "frame_count": 5},
        {"all_targets": True, "selected_track": True},
    ],
)
def test_invalid_clear_never_broadens_scope(http, video, managed_editor, args):
    editor, calls = managed_editor
    response = edit(http, video, editor, args)
    assert response.status_code == 422, response.text
    assert not calls


def test_selected_track_and_stale_session(http, video, managed_editor):
    editor, calls = managed_editor
    response = edit(http, video, editor, {"selected_track": True})
    assert response.status_code == 200, response.text
    assert response.json()["data"]["client_id"] == 10
    assert (
        edit(http, video, editor, {"all_targets": True}, context={"base_revision": 7}).status_code
        == 409
    )
    with http.app.state.services.store.transaction() as session:
        session.update(editor.model_copy(update={"submitted_annotation_id": "submitted"}))
    assert edit(http, video, editor, {"all_targets": True}).status_code == 409
    assert not calls


def test_reviewer_can_edit_review_draft_only(http, client, video, managed_editor):
    editor, calls = managed_editor
    user = client.post(
        "/api/auth/users", {"username": "reviewer", "password": "test-password-1234"}
    )
    client.request(
        "PUT",
        f"/api/projects/{video['project_id']}/members",
        {"user_id": user["id"], "roles": ["reviewer"]},
    )
    client.post("/api/auth/login", {"username": "reviewer", "password": "test-password-1234"})
    assert edit(http, video, editor, {"all_targets": True}).status_code == 403
    with http.app.state.services.store.transaction() as session:
        session.update(editor.model_copy(update={"mode": "review"}))
    for tool, args in [
        ("clear_video_annotations", {"all_targets": True}),
        ("viewer_edit", {"operation": "undo"}),
    ]:
        response = edit(http, video, editor, args, tool=tool)
        assert response.status_code == 200, response.text
    assert not calls
    assert http.app.state.services.store.get(VideoEditor, editor.id).base_revision == 0
