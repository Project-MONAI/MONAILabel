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

"""Guard source ownership and license expressions in the combined distribution."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_package", Path(__file__).resolve().parents[1] / "scripts/build_package.py"
)
assert spec is not None and spec.loader is not None
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_shared_namespace_never_overwrites_a_component(tmp_path):
    build.write_file(tmp_path, "monailabel/core/resource.json", b"first component")
    with pytest.raises(FileExistsError):
        build.write_file(tmp_path, "monailabel/core/resource.json", b"second component")
    assert (tmp_path / "monailabel/core/resource.json").read_bytes() == b"first component"


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py"])
def test_package_files_stay_in_the_staging_directory(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid archive path"):
        build.write_file(tmp_path, name, b"code")


def test_combining_licenses_retains_notices_and_license_choices():
    assert (
        build.combined_license(
            ["Apache-2.0", "Apache-2.0 AND LicenseRef-Slicer", "Apache-2.0 AND MIT"]
        )
        == "Apache-2.0 AND LicenseRef-Slicer AND MIT"
    )
    assert build.combined_license(["Apache-2.0", "MIT OR BSD-3-Clause"]) == (
        "(MIT OR BSD-3-Clause) AND Apache-2.0"
    )
