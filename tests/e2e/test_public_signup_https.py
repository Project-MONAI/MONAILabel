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

"""Automatic signup through the real loopback TLS edge, using disposable accounts."""

import os
import time
from pathlib import Path

import httpx
import pytest
from conftest import VideoStack
from https_fixture import https_edge

pytestmark = pytest.mark.desktop_e2e


def test_public_signup_https_and_cached_starter(tmp_path, monkeypatch):
    if not os.environ.get("MONAILABEL_TEST_NGINX_IMAGE"):
        pytest.skip("Requires an already installed test Nginx image.")
    for flag in (
        "SELF_REGISTER",
        "PUBLIC_REGISTER",
        "CACHE_ONLY",
        "TRAINING_PREVIEW",
        "HTTP_LIMITS",
        "LIFECYCLE",
    ):
        monkeypatch.setenv("MONAILABEL_DEMO_" + flag, "1")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "1")
    cache = os.environ.get("MONAILABEL_TEST_STARTER_CACHE")
    if cache:
        assert Path(cache).is_dir()
        monkeypatch.setenv("MONAILABEL_DATASETS_DIR", cache)
    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url, trust_env=False) as local:
            local.post(
                "/api/auth/setup",
                json={
                    "username": stack.username,
                    "password": stack.password,
                },
            ).raise_for_status()
            foreign = local.post("/api/projects", json={"name": "Host private project"}).json()
        port = int(stack.url.rsplit(":", 1)[-1])
        with (
            https_edge(tmp_path / "edge", port) as edge,
            httpx.Client(
                base_url=edge.url,
                verify=edge.context,
                trust_env=False,
                timeout=30,
            ) as browser,
        ):
            assert browser.get("/api/auth/status").json()["demo_registration"]["automatic_approval"]
            assert browser.get("/api/auth/accounts").status_code == 401
            credentials = {"username": "public-one", "password": "test-password-1234"}
            assert (
                browser.post(
                    "/api/auth/signup",
                    json=credentials,
                    headers={"Origin": "https://other.invalid"},
                ).status_code
                == 403
            )
            signup = browser.post("/api/auth/signup", json=credentials)
            assert signup.status_code == 201 and not signup.json()["is_admin"]
            assert "set-cookie" not in signup.headers
            login = browser.post("/api/auth/login", json=credentials)
            assert login.status_code == 200
            cookie = login.headers["set-cookie"].lower()
            assert all(value in cookie for value in ("secure", "httponly", "samesite=strict"))
            projects = browser.get("/api/projects").json()
            assert len(projects) == 1 and projects[0]["id"] != foreign["id"]
            project = projects[0]["id"]
            assert browser.get(f"/api/projects/{foreign['id']}").status_code == 403
            assert browser.get("/api/auth/accounts").status_code == 403
            assert browser.post("/api/projects", json={"name": "Denied"}).status_code == 403
            assert set(browser.get(f"/api/projects/{project}/permissions").json()["roles"]) == {
                "annotator",
                "reviewer",
            }
            if cache:
                # Only a pre-verified cached public dataset is read; never download.
                for _ in range(240):
                    jobs = browser.get(f"/api/projects/{project}/jobs").json()
                    assert len(jobs) == 1, jobs
                    job = jobs[0]
                    if job["status"] in {"succeeded", "failed", "cancelled"}:
                        break
                    time.sleep(0.25)
                assert job["status"] == "succeeded", job
                assert job["result"]["failed"] == [], job
                assets = browser.get(f"/api/projects/{project}/assets").json()
                assert len(assets) == 1 and assets[0]["annotation_id"] is None
                assert browser.post("/api/demo-access/starter").json() is None
            waiting = {**credentials, "username": "public-waiting"}
            assert browser.post("/api/auth/signup", json=waiting).status_code == 201
            browser.cookies.clear()
            response = browser.post("/api/auth/login", json=waiting)
            assert response.status_code == 429 and response.json()["code"] == "demo_busy"
            # A separate source is intentional: login and signup share a bounded edge bucket.
            with httpx.Client(
                base_url=edge.url,
                verify=edge.context,
                trust_env=False,
                transport=httpx.HTTPTransport(verify=edge.context, local_address="127.0.0.2"),
            ) as admin:
                assert (
                    admin.post(
                        "/api/auth/login",
                        json={
                            "username": stack.username,
                            "password": stack.password,
                        },
                    ).status_code
                    == 200
                )
                directory = admin.get("/api/auth/accounts")
                assert (
                    directory.status_code == 200
                    and "no-store" in directory.headers["cache-control"]
                )
                assert {row["username"] for row in directory.json()} == {
                    stack.username,
                    "public-one",
                    "public-waiting",
                }
                for route in ("users", "demo-users", "setup", "token", "register-demo"):
                    assert admin.post("/api/auth/" + route).status_code == 404
            for _ in range(10):
                response = browser.post("/api/auth/signup", json=credentials)
                if response.status_code == 429:
                    break
            assert response.status_code == 429 and response.headers.get("retry-after")
    finally:
        stack.stop_workspace()
