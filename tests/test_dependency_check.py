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

"""The ARM64 vendor-tag exception must never hide incompatible dependencies."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "check_dependencies", Path(__file__).resolve().parents[1] / "scripts/check_dependencies.py"
)
assert spec is not None and spec.loader is not None
check_dependencies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_dependencies)

WARNING = "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"


@pytest.fixture
def vendor_wheel(tmp_path, monkeypatch):
    library = tmp_path / "libcusparseLt.so.0"
    library.write_bytes(b"\x7fELF\x02\x01" + bytes(12) + b"\xb7\x00")
    package = SimpleNamespace(
        version="0.8.0",
        read_text=lambda _: "Wheel-Version: 1.0\nTag: py3-none-manylinux2014_sbsa\n",
        locate_file=lambda _: library,
    )
    monkeypatch.setattr(check_dependencies.platform, "system", lambda: "Linux")
    monkeypatch.setattr(check_dependencies.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(check_dependencies, "distribution", lambda _: package)
    return library, package


def test_accepts_only_the_verified_arm64_vendor_tag(vendor_wheel):
    assert check_dependencies.known_sbsa_tag_warning(WARNING)
    library, _ = vendor_wheel
    library.write_bytes(b"\x7fELF\x02\x01" + bytes(12) + b"\x3e\x00")
    assert not check_dependencies.known_sbsa_tag_warning(WARNING)


@pytest.mark.parametrize(
    "output",
    [WARNING + "\ntorch requires a missing dependency", WARNING.replace("0.8.0", "0.8.1")],
)
def test_other_dependency_errors_are_not_exempted(vendor_wheel, output):
    assert not check_dependencies.known_sbsa_tag_warning(output)


def test_other_platforms_and_metadata_are_not_exempted(vendor_wheel, monkeypatch):
    _, package = vendor_wheel
    package.read_text = lambda _: "Tag: py3-none-manylinux2014_x86_64\n"
    assert not check_dependencies.known_sbsa_tag_warning(WARNING)
    monkeypatch.setattr(check_dependencies.platform, "machine", lambda: "x86_64")
    assert not check_dependencies.known_sbsa_tag_warning(WARNING)
