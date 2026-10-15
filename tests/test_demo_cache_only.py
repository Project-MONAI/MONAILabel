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

from types import SimpleNamespace

import pytest

from monailabel.core.errors import DomainError
from monailabel.server.dataset_downloads import Downloads, sources


def test_cache_only_never_downloads_or_creates_missing_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_CACHE_ONLY", "1")
    downloads = Downloads(tmp_path / "not-created")
    source = next(source for source in sources() if source.importable)
    with pytest.raises(DomainError, match="not prepared"):
        downloads.fetch(source, SimpleNamespace(progress=lambda *_: None))
    assert not downloads.root.exists()


def test_catalog_and_direct_import_reject_uncached_datasets_before_job(
    http, client, seeded, monkeypatch
):
    monkeypatch.setenv("MONAILABEL_DEMO_CACHE_ONLY", "1")
    service = http.app.state.services
    assert service.dataset_templates.catalog() == []
    source = next(source for source in sources() if source.importable)
    from monailabel.core.dataset_templates import DatasetTemplateImport

    with pytest.raises(DomainError, match="not prepared"):
        service.dataset_templates.start(
            seeded[0]["project_id"], DatasetTemplateImport(template_id=source.id), "fixture"
        )
    assert client.get(f"/api/projects/{seeded[0]['project_id']}/jobs") == []


def test_cache_only_accepts_verified_file_without_network_or_rehash(tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_CACHE_ONLY", "1")
    downloads = Downloads(tmp_path)
    source = next(source for source in sources() if source.importable)
    # Synthetic marker fixture; the normal preparation path verifies the catalog checksum.
    path = downloads.path(source)
    path.write_bytes(b"synthetic archive")
    downloads._remember(source)
    assert downloads.fetch(source, SimpleNamespace(progress=lambda *_: None)) == path
    path.write_bytes(b"changed")
    with pytest.raises(DomainError, match="not prepared"):
        downloads.fetch(source, SimpleNamespace(progress=lambda *_: None))
