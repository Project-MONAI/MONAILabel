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

import json
from datetime import UTC, datetime, timedelta

from monailabel.server.demo_audit import DemoAudit


def test_disabled_does_not_create_files(tmp_path):
    audit = DemoAudit(tmp_path / "audit")
    audit.record("assistant.request", "visitor", message="hello")
    audit.close()
    assert not audit.root.exists()


def test_payloads_secrets_permissions_and_rotation(tmp_path):
    audit = DemoAudit(tmp_path / "audit", enabled=True)
    audit.max_bytes = 1000
    for _ in range(30):
        audit.record("assistant.request", "visitor", message="x" * 200)
    audit.record("assistant.request", "visitor", message="password=never-record-this")
    audit.queue.join()
    audit.close()
    files = list(audit.root.glob("*.jsonl"))
    assert 1 <= len(files) <= 4
    assert sum(path.stat().st_size for path in files) <= 4000
    content = "".join(path.read_text() for path in files)
    assert "never-record-this" not in content
    assert "omitted: possible credential" in content
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)
    assert audit.root.stat().st_mode & 0o777 == 0o700


def test_writer_failure_and_rate_limit_are_counted_not_raised(tmp_path, monkeypatch):
    audit = DemoAudit(tmp_path / "audit", enabled=True)

    def fail(_):
        raise OSError("disk full")

    monkeypatch.setattr(audit, "_write", fail)
    for _ in range(500):
        audit.record("ui.click", "visitor", target="models")
    audit.queue.join()
    audit.close()
    status = audit.status()
    assert status["dropped"] == 500
    assert status["write_errors"] > 0


def test_retention_uses_event_age_not_recent_file_modification(tmp_path):
    audit = DemoAudit(tmp_path / "audit")
    audit.root.mkdir()
    old = datetime.now(UTC) - timedelta(days=8)
    path = audit._path(0)
    path.write_text(json.dumps({"at": old.isoformat()}) + "\n")
    audit._prune()
    assert not path.exists()


def test_event_api_never_accepts_form_values_or_free_text(http):
    assert (
        http.post("/api/demo-access/events", json={"kind": "click", "target": "models"}).status_code
        == 204
    )
    assert (
        http.post(
            "/api/demo-access/events",
            json={"kind": "click", "target": "models", "password": "secret"},
        ).status_code
        == 422
    )
    assert (
        http.post(
            "/api/demo-access/events", json={"kind": "click", "target": "some patient name"}
        ).status_code
        == 422
    )
    assert http.get("/api/demo-access/audit-status").json()["enabled"] is False
