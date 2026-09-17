"""Archive downloads persist independently of import selections and server instances."""

import hashlib
import io
import json
import os
import tarfile
import zipfile
from unittest.mock import Mock

import httpx
import pytest
from filelock import FileLock

from monailabel.core.errors import Cancelled, DomainError
from monailabel.server.dataset_downloads import Downloads, Source, dataset_cache_dir
from monailabel.server.jobs import JobContext


@pytest.fixture(params=["tar", "zip"])
def download(tmp_path, monkeypatch, request):
    buffer = io.BytesIO()
    data = b"dataset contents"
    if request.param == "tar":
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            item = tarfile.TarInfo("image.bin")
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    else:
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("image.bin", data)
    payload = buffer.getvalue()
    source = Source(
        id="sample",
        name="Sample",
        category="Radiology",
        description="Test fixture",
        source_url="https://datasets.invalid/",
        license="Test",
        url=f"https://datasets.invalid/sample.{request.param}",
        download_bytes=len(payload),
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        format="msd",
    )
    requests = []

    def respond(incoming):
        requests.append(incoming)
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        yield Downloads(tmp_path / "dataset-downloads"), source, payload, requests


def test_completed_archive_reused_by_fresh_cache_instance(download):
    cache, source, payload, requests = download
    path = cache.fetch(source, Mock(spec=JobContext))
    assert path.read_bytes() == payload
    assert len(requests) == 1
    # A fresh service has no in-memory knowledge of the previous import.
    restarted = Downloads(cache.root)
    assert restarted.cached(source)
    assert restarted.fetch(source, Mock(spec=JobContext)) == path
    assert cache.fetch(source, Mock(spec=JobContext)) == path
    assert len(requests) == 1
    assert path.read_bytes() == payload


def test_cache_location_respects_workspace_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("MONAILABEL_DATASETS_DIR")
    monkeypatch.delenv("MONAILABEL_CACHE_DIR", raising=False)
    monkeypatch.setenv("MONAILABEL_DATA_DIR", str(tmp_path / "workspace"))
    assert dataset_cache_dir() == tmp_path / "workspace" / ".cache" / "datasets"
    assert dataset_cache_dir(tmp_path / "other") == tmp_path / "other" / ".cache" / "datasets"
    monkeypatch.setenv("MONAILABEL_CACHE_DIR", str(tmp_path / "common"))
    assert dataset_cache_dir() == tmp_path / "common" / "datasets"
    monkeypatch.setenv("MONAILABEL_DATASETS_DIR", str(tmp_path / "specific"))
    assert dataset_cache_dir() == tmp_path / "specific"


@pytest.mark.parametrize("copy_required", [False, True])
def test_legacy_archive_shared_across_workspaces_without_redownload(
    download, monkeypatch, copy_required
):
    cache, source, payload, requests = download
    original = cache.fetch(source, Mock(spec=JobContext))
    original.with_suffix(".verified.json").unlink()
    if copy_required:
        monkeypatch.setattr(os, "link", Mock(side_effect=OSError("Different filesystem")))
    shared_root = cache.root.parent / "shared"
    first_workspace = Downloads(shared_root, legacy_root=cache.root)
    path = first_workspace.fetch(source, Mock(spec=JobContext))
    assert path.parent == shared_root
    assert path.read_bytes() == payload
    assert original.read_bytes() == payload
    assert first_workspace.cached(source)
    second_workspace = Downloads(shared_root, legacy_root=cache.root.parent / "another-workspace")
    assert second_workspace.fetch(source, Mock(spec=JobContext)) == path
    assert len(requests) == 1


@pytest.mark.parametrize("marker", ["missing", "invalid", "legacy"])
def test_saved_archive_recovered_without_network(download, marker):
    cache, source, payload, requests = download
    cache.root.mkdir()
    path = cache.path(source)
    path.write_bytes(payload)
    if marker != "missing":
        path.with_suffix(".verified.json").write_text(
            "broken JSON"
            if marker == "invalid"
            else json.dumps({"checksum": source.checksum, "size": len(payload)})
        )
    assert not cache.cached(source)
    assert cache.fetch(source, Mock(spec=JobContext)) == path
    assert cache.cached(source)
    assert requests == []


def test_modified_archive_revalidated_then_replaced(download):
    cache, source, payload, requests = download
    path = cache.fetch(source, Mock(spec=JobContext))
    old_stat = path.stat()
    path.write_bytes(b"X" * len(payload))  # Same size must not bypass integrity checks.
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000))
    assert not cache.cached(source)
    assert cache.fetch(source, Mock(spec=JobContext)) == path
    assert len(requests) == 2
    assert path.read_bytes() == payload
    assert cache.cached(source)


def test_checksum_failure_retains_previous_archive(download):
    cache, source, payload, requests = download
    path = cache.fetch(source, Mock(spec=JobContext))
    changed = source.model_copy(update={"checksum": "sha256:" + "0" * 64})
    with pytest.raises(DomainError, match="checksum"):
        cache.fetch(changed, Mock(spec=JobContext))
    assert len(requests) == 2
    assert path.read_bytes() == payload
    assert cache.cached(source)
    assert not cache.cached(changed)
    assert not path.with_suffix(path.suffix + ".part").exists()


def test_lock_prevents_duplicate_download_and_cancel_never_marks_partial_complete(download):
    cache, source, _, requests = download
    cache.root.mkdir()
    path = cache.path(source)
    with FileLock(str(path) + ".lock"), pytest.raises(DomainError, match="already downloading"):
        cache.fetch(source, Mock(spec=JobContext))
    assert requests == []
    context = Mock(spec=JobContext)
    context.progress.side_effect = Cancelled()
    with pytest.raises(Cancelled):
        cache.fetch(source, context)
    assert not cache.cached(source)
    assert not path.exists()
    assert not path.with_suffix(path.suffix + ".part").exists()
