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

"""Offline guards only; never execute the administrator installation path."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_host_bootstrap.sh"


class DemoHostBootstrapTests(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            for command in (
                "sudo",
                "apt-get",
                "curl",
                "docker",
                "systemctl",
                "loginctl",
                "usermod",
                "nvidia-ctk",
                "nvidia-smi",
                "install",
                "runuser",
            ):
                blocker = Path(directory) / command
                blocker.write_text("#!/bin/sh\necho UNEXPECTED_MUTATION >&2\nexit 99\n")
                blocker.chmod(0o700)
            result = subprocess.run(
                ["/bin/bash", str(SCRIPT), *args],
                env={**os.environ, "PATH": f"{directory}:/usr/bin:/bin"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotIn("UNEXPECTED_MUTATION", result.stdout + result.stderr)
            return result

    def test_default_and_help_are_read_only(self) -> None:
        for args in ((), ("--plan",), ("--help",)):
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("ROOT-EQUIVALENT", result.stdout)

    def test_apply_requires_explicit_privilege_acknowledgment(self) -> None:
        result = self.run_script("--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("allow-docker-root-access", result.stderr)

    def test_unknown_options_fail_without_changes(self) -> None:
        result = self.run_script("--unknown")
        self.assertNotEqual(result.returncode, 0)

    def test_unprotected_source_cannot_install(self) -> None:
        result = self.run_script("--apply", "--allow-docker-root-access")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("copy", result.stderr)


if __name__ == "__main__":
    unittest.main()
