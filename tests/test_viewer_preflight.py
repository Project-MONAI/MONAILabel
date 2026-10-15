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

"""Prepared-only viewer checks do not build, download, or change shared services."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from monailabel.core.errors import DomainError
from monailabel.viewers.browser_desktop import BrowserDesktop
from monailabel.viewers.cvat_runtime import VERSION, CvatManager
from monailabel.viewers.ohif import OhifManager
from monailabel.viewers.provisioning.desktop import DesktopProvisioner


@pytest.mark.parametrize("image,client", [(False, False), (True, False), (False, True)])
def test_browser_preflight_rejects_missing_resources_without_installing(
    tmp_path, monkeypatch, image, client
):
    runtime = BrowserDesktop(tmp_path / "sessions", "fixture", tmp_path / "tools")
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("shutil.which", lambda _: "/test/docker")
    forbidden = Mock(side_effect=AssertionError("No download or process creation allowed"))
    monkeypatch.setattr(httpx, "get", forbidden)
    if client:
        runtime.dist.mkdir(parents=True)
        (runtime.dist / "vnc.html").write_text("Prepared test display client")
    before = set(tmp_path.rglob("*"))

    def inspect(*args, **kwargs):
        assert args[:3] == ("image", "ls", "-q")
        return "fixture-image" if image else ""

    monkeypatch.setattr(runtime, "run", inspect)
    with pytest.raises(DomainError, match="not prepared"):
        runtime.prepare(lambda _: None, download=False)
    assert set(tmp_path.rglob("*")) == before
    forbidden.assert_not_called()


def test_browser_preflight_requires_native_installation_without_creating_cache(
    tmp_path, monkeypatch
):
    runtime = BrowserDesktop(tmp_path / "sessions", "fixture", tmp_path / "tools")
    monkeypatch.setattr(runtime, "prepare", lambda *args, **kwargs: "fixture-image")
    monkeypatch.setattr(DesktopProvisioner, "discover", lambda *args: None)
    with pytest.raises(DomainError, match="not installed or cached"):
        runtime.preflight("slicer")
    assert not runtime.root.exists() and not runtime.viewers.root.exists()


def test_ohif_requires_current_build_without_invoking_node(tmp_path, monkeypatch):
    monkeypatch.delenv("MONAILABEL_OHIF_DIST", raising=False)
    manager = OhifManager(tmp_path / "tools")
    build = Mock()
    monkeypatch.setattr("monailabel.viewers.ohif.subprocess.run", build)
    with pytest.raises(DomainError, match="not prepared"):
        manager.ensure(download=False)
    assert not manager.root.exists()
    build.assert_not_called()

    # Simulate host preparation, then use the real signature/manifest checks.
    def host_build(*args, **kwargs):
        manager.dist.mkdir(parents=True)
        (manager.dist / "index.html").write_text("Prepared test viewer")
        return SimpleNamespace(returncode=0)

    build.side_effect = host_build
    monkeypatch.setattr("monailabel.viewers.ohif.shutil.which", lambda _: "/test/node")
    assert manager.ensure() == manager.dist
    before = {p: p.read_bytes() for p in manager.root.rglob("*") if p.is_file()}
    assert manager.ensure(download=False) == manager.dist
    assert {p: p.read_bytes() for p in before} == before
    assert build.call_count == 1
    (manager.source.parent / "build.json").write_text('{"extension_hash":"old-build"}')
    with pytest.raises(DomainError, match="not prepared"):
        manager.ensure(download=False)
    assert build.call_count == 1


@pytest.mark.parametrize("ready", [False, True])
def test_cvat_prepared_only_never_starts_or_modifies_services(tmp_path, monkeypatch, ready):
    manager = CvatManager(tmp_path, "fixture")
    commands = Mock(side_effect=AssertionError("No Docker/Compose changes allowed"))
    monkeypatch.setattr(manager, "_run", commands)
    calls = []
    original = httpx.Client

    def about(request):
        calls.append(request)
        assert str(request.url) == "http://127.0.0.1:8123/api/server/about"
        return httpx.Response(200, json={"version": VERSION})

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        return original(transport=httpx.MockTransport(about), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    if ready:
        manager.dist.mkdir(parents=True)
        (manager.root / "runtime.json").write_text(json.dumps({"port": 8123, "version": VERSION}))
        (manager.dist / "index.html").write_text("Prepared test viewer")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if ready:
        assert manager.ensure("unused", "unused", lambda _: None, download=False) == (
            "http://127.0.0.1:8123"
        )
        assert len(calls) == 1
    else:
        with pytest.raises(DomainError, match="not prepared"):
            manager.ensure("unused", "unused", lambda _: None, download=False)
        assert not calls and not manager.root.exists()
    commands.assert_not_called()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("failure", ["version", "unavailable"])
def test_cvat_preflight_fails_closed_for_stale_or_stopped_service(tmp_path, monkeypatch, failure):
    manager = CvatManager(tmp_path, "fixture")
    manager.dist.mkdir(parents=True)
    (manager.dist / "index.html").touch()
    (manager.root / "runtime.json").write_text(json.dumps({"port": 8123, "version": VERSION}))
    original = httpx.Client

    def about(request):
        if failure == "unavailable":
            raise httpx.ConnectError("Stopped test service")
        return httpx.Response(200, json={"version": "old-version"})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(about), **kwargs)
    )
    with pytest.raises(DomainError, match="not prepared"):
        manager.prepared()
