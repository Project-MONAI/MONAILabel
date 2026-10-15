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

"""Offline verification and no-overwrite guards for resumable public downloads."""

import hashlib
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/demo_prefetch.py"


@pytest.mark.parametrize("existing", [True, False])
def test_fetch_verifies_before_publishing(tmp_path, monkeypatch, existing):
    module = runpy.run_path(str(SCRIPT))
    content = b"pinned public data"
    target = tmp_path / "cached"
    if existing:
        target.write_bytes(content)
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        assert "--continue-at" in command and "=https" in command
        Path(command[command.index("--output") + 1]).write_bytes(content)

    monkeypatch.setattr(module["subprocess"], "run", download)
    result = module["fetch"](
        "https://example.invalid/pinned", target, "sha256", hashlib.sha256(content).hexdigest()
    )
    assert result == target and target.read_bytes() == content
    assert len(calls) == (0 if existing else 1)
    assert not target.with_name("cached.resume-part").exists()


@pytest.mark.parametrize("existing", [True, False])
def test_checksum_mismatch_preserves_evidence(tmp_path, monkeypatch, existing):
    module = runpy.run_path(str(SCRIPT))
    target = tmp_path / "cached"
    corrupt = b"not the pinned content"
    if existing:
        target.write_bytes(corrupt)

    def download(command, **kwargs):
        assert not existing, "An existing file must never be overwritten."
        Path(command[command.index("--output") + 1]).write_bytes(corrupt)

    monkeypatch.setattr(module["subprocess"], "run", download)
    with pytest.raises(RuntimeError, match="checksum"):
        module["fetch"]("https://example.invalid/pinned", target, "sha256", "0" * 64)
    if existing:
        assert target.read_bytes() == corrupt
    else:
        assert not target.exists()
        assert target.with_name("cached.resume-part").read_bytes() == corrupt
