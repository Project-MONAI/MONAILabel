"""Project maintenance and browser resource contracts."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_workspace_pages_support_direct_load_without_hiding_missing_resources(http):
    index = http.get("/")
    for page in ("datasets", "models", "reviews", "activity", "team"):
        response = http.get(f"/{page}")
        assert response.status_code == 200
        assert response.text == index.text
        assert "text/html" in response.headers["content-type"]
        assert http.get(f"/{page}/").text == index.text
    for path in ("/missing", "/api/missing", "/static/missing.js"):
        response = http.get(path)
        assert response.status_code == 404
        assert response.content != index.content
    assert http.get("/ohif/missing").content != index.content


def test_project_settings_preserve_protocol_and_require_current_version(http, client, seeded):
    setup, assets = seeded
    path = f"/api/projects/{setup['project_id']}"
    project = client.get(path)
    body = {
        "name": "Renamed project",
        "instructions": "Review the full volume.",
        "base_version": project["version"],
    }
    response = http.patch(path, json=body)
    assert response.status_code == 200
    updated = response.json()
    assert updated["name"] == body["name"]
    assert updated["version"] == project["version"] + 1
    for field in ("labels", "defaults", "protocol_version"):
        assert updated[field] == project[field]
    assert client.get(path + "/assets") == assets
    assert http.patch(path, json=body).status_code == 409


def test_managers_assign_roles_by_username_without_exposing_all_users(http, client, seeded):
    setup, _ = seeded
    path = f"/api/projects/{setup['project_id']}"
    for name in ("manager", "reviewer"):
        user = client.post("/api/auth/users", {"username": name, "password": "test-password-1234"})
        if name == "manager":
            client.request("PUT", path + "/members", {"user_id": user["id"], "roles": [name]})
    http.post("/api/auth/login", json={"username": "manager", "password": "test-password-1234"})
    assert http.get("/api/auth/users").status_code == 403
    response = http.put(path + "/members", json={"username": " Reviewer ", "roles": ["reviewer"]})
    assert response.status_code == 200
    members = client.get(path + "/members")
    assert next(m for m in members if m["username"] == "reviewer")["active"]
    http.post("/api/auth/login", json={"username": "reviewer", "password": "test-password-1234"})
    assert http.get(path + "/members").status_code == 403
    assert http.patch(path, json={"name": "Denied", "base_version": 0}).status_code == 403


def test_shared_speech_resource_is_packaged_and_served_under_script_policy(http):
    response = http.get("/static/speech.js")
    assert response.status_code == 200
    assert "text/javascript" in response.headers["content-type"]
    assert "export function createSpeech" in response.text
    assert "script-src 'self'" in response.headers["content-security-policy"]


def test_browser_speech_lifecycle():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the shared browser speech contract check.")
    subprocess.run([node, str(Path(__file__).parent / "web/speech.mjs")], check=True)


def test_project_default_is_revision_checked_and_unsupported_organs_do_not_fall_back(
    http, client, seeded
):
    setup, assets = seeded
    path = f"/api/projects/{setup['project_id']}"
    project = client.get(path)
    labels = [label["id"] for label in project["labels"] if label["id"]]
    model = client.post(
        path + "/models",
        {
            "name": "Single structure",
            "provider": "http-mask",
            "label_ids": [0, labels[0]],
            "config": {"url": "http://127.0.0.1:9999/infer"},
        },
    )
    body = {"model_id": model["id"], "base_version": project["version"]}
    response = http.put(path + "/annotation-model", json=body)
    assert response.status_code == 200
    assert response.json()["annotation_model_id"] == model["id"]
    assert response.json()["defaults"] == {str(labels[0]): model["id"]}
    assert http.put(path + "/annotation-model", json=body).status_code == 409
    response = http.post(f"/api/assets/{assets[0]['id']}/annotate", json={"label_ids": [labels[1]]})
    assert response.status_code == 422 and "does not support" in response.json()["detail"]
    assert client.get(path + "/jobs") == []
    user = client.post(
        "/api/auth/users", {"username": "annotator", "password": "test-password-1234"}
    )
    client.request("PUT", path + "/members", {"user_id": user["id"], "roles": ["annotator"]})
    client.post("/api/auth/login", {"username": "annotator", "password": "test-password-1234"})
    assert http.put(path + "/annotation-model", json=body).status_code == 403


def test_training_catalog_limits_demo_recipe_to_demo_projects(client, seeded):
    demo, _ = seeded
    project = client.post("/api/projects", {"name": "Annotation project"})
    recipes = client.get(f"/api/projects/{project['id']}/recipes")
    assert {recipe["id"] for recipe in recipes} == {"monai-unet", "vista3d"}
    assert all(not recipe["demo_only"] for recipe in recipes)
    demo_recipes = client.get(f"/api/projects/{demo['project_id']}/recipes")
    assert next(recipe for recipe in demo_recipes if recipe["id"] == "pixel-gaussian")["demo_only"]
