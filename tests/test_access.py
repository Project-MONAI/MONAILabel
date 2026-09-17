import pytest

from monailabel.core.models import User


def sign_in(http, username):
    response = http.post(
        "/api/auth/login", json={"username": username, "password": "test-password-1234"}
    )
    assert response.status_code == 200


def test_authentication_and_origin_boundary(http):
    assert (
        http.post(
            "/api/auth/setup", json={"username": "second", "password": "test-password-1234"}
        ).status_code
        == 409
    )
    assert (
        http.post("/api/demo", headers={"Origin": "https://malicious.example"}).status_code == 403
    )
    assert http.get("/api/projects", headers={"host": "malicious.example"}).status_code == 400
    http.post("/api/auth/logout")
    assert http.get("/api/projects").status_code == 401
    assert http.post("/api/demo").status_code == 401
    assert (
        http.post(
            "/api/auth/login", json={"username": "owner", "password": "incorrect-password"}
        ).status_code
        == 401
    )


@pytest.mark.parametrize("role", ["annotator", "reviewer"])
def test_project_roles_are_enforced_everywhere(http, client, seeded, role):
    setup, assets = seeded
    pid, aid = setup["project_id"], assets[0]["id"]
    own = client.get("/api/auth/me")
    user = client.post("/api/auth/users", {"username": role, "password": "test-password-1234"})
    client.request("PUT", f"/api/projects/{pid}/members", {"user_id": user["id"], "roles": [role]})
    other = client.post(
        "/api/projects",
        {
            "name": "Other project",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Liver", "color": "#ff0000"},
            ],
        },
    )
    annotation = client.post(
        f"/api/assets/{aid}/review",
        {
            "base_revision": 0,
            "mask": client.get(f"/api/assets/{aid}/fixture")["mask"],
            "covered_labels": [0, 1, 2],
        },
    )
    sign_in(http, role)
    assert [p["id"] for p in client.get("/api/projects")] == [pid]
    assert http.get(f"/api/projects/{other['id']}/assets").status_code == 403
    assert http.get(f"/api/projects/{pid}/credentials").status_code == 403
    assert http.post(f"/api/projects/{pid}/snapshots").status_code == 403
    assert (
        http.post(f"/api/projects/{pid}/assistant", json={"message": "create snapshot"}).status_code
        == 403
    )
    assert (
        http.post(
            "/api/assistant", json={"message": "configure models", "project_id": pid}
        ).status_code
        == 403
    )
    status = http.post(
        f"/api/annotations/{annotation['id']}/decision", json={"verdict": "accepted"}
    ).status_code
    assert status == (201 if role == "reviewer" else 403)
    status = http.post(f"/api/assets/{aid}/annotate", json={}).status_code
    assert status == (202 if role == "annotator" else 403)
    status = http.post(
        f"/api/assets/{aid}/review-mask",
        params={"base_revision": 1, "covered_labels": [0, 1, 2]},
        content=b"bad-size",
    ).status_code
    assert status == (422 if role == "annotator" else 403)
    status = http.post(
        f"/api/projects/{pid}/assistant",
        json={"message": "annotate this volume", "context": {"asset_id": aid}},
    ).status_code
    assert status == (200 if role == "annotator" else 403)
    assert (
        http.post(
            "/api/auth/users", json={"username": "escalate", "password": "test-password-1234"}
        ).status_code
        == 403
    )
    assert (
        http.put(
            f"/api/projects/{pid}/members", json={"user_id": user["id"], "roles": ["manager"]}
        ).status_code
        == 403
    )
    assert client.get("/api/auth/me")["id"] != own["id"]


def test_keys_are_private_rotatable_and_scoped(http, client, seeded):
    setup, assets = seeded
    pid = setup["project_id"]
    secret = "secret-value-do-not-leak-12345"
    cred = client.post(f"/api/projects/{pid}/credentials", {"name": "Inference", "api_key": secret})
    assert secret not in str(cred)
    model = client.post(
        f"/api/projects/{pid}/models",
        {
            "name": "Mask service",
            "provider": "http-mask",
            "label_ids": [0, 1, 2],
            "config": {"url": "https://example.org/predict", "credential_id": cred["id"]},
        },
    )
    assert secret not in str(model)
    service = http.app.state.services
    assert service.secrets.resolve(pid, cred["id"]) == secret
    assert secret.encode() not in service.store.path.read_bytes()
    invalid = http.post(f"/api/projects/{pid}/credentials", json={"name": "", "api_key": secret})
    assert invalid.status_code == 422 and secret not in invalid.text
    rotated = client.post(
        f"/api/projects/{pid}/credentials",
        {"name": "Rotated", "api_key": "replacement-secret", "credential_id": cred["id"]},
    )
    assert rotated["id"] == cred["id"]
    assert service.secrets.resolve(pid, cred["id"]) == "replacement-secret"
    other = client.post("/api/demo")["project_id"]
    assert (
        http.post(
            f"/api/projects/{other}/models",
            json={
                "name": "Wrong key scope",
                "provider": "http-mask",
                "label_ids": [0, 1, 2],
                "config": {"url": "https://example.org/predict", "credential_id": cred["id"]},
            },
        ).status_code
        == 403
    )
    user = client.post(
        "/api/auth/users", {"username": "disabled", "password": "test-password-1234"}
    )
    client.request("PUT", f"/api/auth/users/{user['id']}", {"active": False})
    assert (
        http.post(
            "/api/auth/login", json={"username": "disabled", "password": "test-password-1234"}
        ).status_code
        == 401
    )


def test_disabled_identity_invalidates_existing_tokens(http, client):
    user = client.post(
        "/api/auth/users", {"username": "temporary", "password": "test-password-1234"}
    )
    token = http.app.state.services.auth.issue(User.model_validate(user))
    client.request("PUT", f"/api/auth/users/{user['id']}", {"active": False})
    assert http.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
