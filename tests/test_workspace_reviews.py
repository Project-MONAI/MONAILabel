"""Workspace review actions preserve history, revisions, permissions and viewer drafts."""

import pytest

from monailabel.core.chat import ChatMessage, ToolCall


def pending(client, seeded):
    setup, assets = seeded
    annotations = []
    for asset in assets[:2]:
        annotations.append(
            client.post(
                f"/api/assets/{asset['id']}/review",
                {
                    "base_revision": 0,
                    "covered_labels": [0, 1, 2],
                    "mask": client.get(f"/api/assets/{asset['id']}/fixture")["mask"],
                },
            )
        )
    return f"/api/projects/{setup['project_id']}", annotations


def test_batch_decisions_reset_history_and_reject_stale_requests(client, http, seeded):
    prefix, annotations = pending(client, seeded)
    route = prefix + "/review-decisions"
    items = [{"annotation_id": a["id"]} for a in annotations]
    accepted = client.post(route, {"items": items, "verdict": "accepted"})
    assert len(accepted) == 2
    # Both annotation and decision revisions are checked, including no-op attempts.
    assert http.post(route, json={"items": items, "verdict": "pending"}).status_code == 409
    items = [{"annotation_id": d["annotation_id"], "decision_id": d["id"]} for d in accepted]
    reset = client.post(route, {"items": items, "verdict": "pending"})
    assert len(reset) == 2 and all(d["verdict"] == "pending" for d in reset)
    assert client.get(prefix + "/decisions") == accepted + reset
    assert http.post(prefix + "/snapshots").status_code == 422
    assert all(
        client.get(f"/api/assets/{a['asset_id']}")["annotation_id"] == a["id"] for a in annotations
    )
    items = [{"annotation_id": d["annotation_id"], "decision_id": d["id"]} for d in reset]
    assert client.post(route, {"items": items, "verdict": "pending"}) == []
    changes = client.post(
        route, {"items": items, "verdict": "changes_requested", "comment": "Edge"}
    )
    assert all(d["comment"] == "Edge" for d in changes)


def test_batch_is_atomic_when_a_later_annotation_was_replaced(client, http, seeded):
    prefix, annotations = pending(client, seeded)
    changed = annotations[1]
    client.post(
        f"/api/assets/{changed['asset_id']}/review",
        {
            "base_revision": changed["revision"],
            "covered_labels": [0, 1, 2],
            "mask": client.get(f"/api/assets/{changed['asset_id']}/fixture")["mask"],
        },
    )
    response = http.post(
        prefix + "/review-decisions",
        json={"items": [{"annotation_id": a["id"]} for a in annotations], "verdict": "accepted"},
    )
    assert response.status_code == 409
    assert client.get(prefix + "/decisions") == []


@pytest.mark.parametrize("role, expected", [("reviewer", 200), ("annotator", 403)])
def test_batch_review_permissions(client, http, seeded, role, expected):
    prefix, annotations = pending(client, seeded)
    user = client.post("/api/auth/users", {"username": role, "password": "test-password-1234"})
    client.request("PUT", prefix + "/members", {"user_id": user["id"], "roles": [role]})
    client.post("/api/auth/login", {"username": role, "password": "test-password-1234"})
    response = http.post(
        prefix + "/review-decisions",
        json={"items": [{"annotation_id": a["id"]} for a in annotations], "verdict": "accepted"},
    )
    assert response.status_code == expected


def test_chat_pending_scope_reset_and_native_viewer_draft_protection(client, http, seeded):
    prefix, annotations = pending(client, seeded)
    planner = http.app.state.services.assistants.provider
    for scope, verdict in [("pending", "accepted"), ("accepted", "pending")]:
        planner.queue = [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=scope,
                        name="review_saved_annotations",
                        arguments={"scope": scope, "verdict": verdict},
                    )
                ],
            )
        ]
        reply = client.post(prefix + "/assistant", {"message": "Update the saved reviews"})
        assert reply["tools"] == ["review_saved_annotations"]
        assert "2 saved reviews" in reply["message"]
    decisions = client.get(prefix + "/decisions")
    assert len(decisions) == 4 and all(d["verdict"] == "pending" for d in decisions[-2:])
    planner.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="viewer-draft-guard",
                    name="review_saved_annotations",
                    arguments={"scope": "all", "verdict": "accepted"},
                )
            ],
        )
    ]
    response = http.post(
        prefix + "/assistant",
        json={
            "message": "Good",
            "context": {"asset_id": annotations[0]["asset_id"], "viewer_actions": ["review"]},
        },
    )
    assert response.status_code == 422
    assert client.get(prefix + "/decisions") == decisions
