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

"""Check dependencies, recognizing NVIDIA's pinned ARM64 cuSPARSELt wheel tag defect."""

import platform
import subprocess
import sys
from importlib.metadata import distribution


def known_sbsa_tag_warning(output: str) -> bool:
    if (
        platform.system() != "Linux"
        or platform.machine() != "aarch64"
        or output.strip() != "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"
    ):
        return False
    package = distribution("nvidia-cusparselt-cu13")
    wheel = package.read_text("WHEEL") or ""
    if package.version != "0.8.0" or "Tag: py3-none-manylinux2014_sbsa" not in wheel.splitlines():
        return False
    # The published filename is aarch64, but WHEEL says sbsa. Check the actual
    # ELF library rather than accepting arbitrary incompatible distributions.
    path = package.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0")
    with path.open("rb") as stream:
        header = stream.read(20)
    return header[:6] == b"\x7fELF\x02\x01" and header[18:20] == b"\xb7\x00"


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check", "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    elif not result.stderr and known_sbsa_tag_warning(result.stdout):
        print("Dependency requirements satisfied; verified NVIDIA ARM64 library has an SBSA tag.")
    else:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()


if __name__ == "__main__":
    main()
