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

"""Offline safety checks, also runnable with system Python before installing dependencies."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "scripts" / "demo_install.sh"


class DemoInstallTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="monailabel-demo-install-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "unexpected-command"
        for name in ("sudo", "apt-get", "curl", "uv", "docker", "systemctl", "usermod", "sh"):
            path = self.bin / name
            path.write_text('#!/bin/bash\nprintf "%s\\n" "$0" >> "$DEMO_TEST_LOG"\nexit 93\n')
            path.chmod(0o700)
        self.environment = {
            **os.environ,
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "DEMO_TEST_LOG": str(self.log),
        }

    def run_installer(self, *args):
        result = subprocess.run(
            ["/bin/bash", str(INSTALLER), *args],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertFalse(self.log.exists(), result.stdout + result.stderr)
        self.assertEqual({p.name for p in self.root.iterdir()}, {"bin"})
        return result

    def test_default_is_read_only(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No changes will be made", result.stdout)

    def test_help_is_read_only(self):
        result = self.run_installer("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("never execute", result.stdout)

    def test_environment_needs_explicit_apply(self):
        result = self.run_installer("environment", "--root", str(self.root / "unused"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No changes will be made", result.stdout)

    def test_unknown_phase_is_rejected(self):
        result = self.run_installer("all", "--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown phase", result.stderr)

    def test_system_apply_is_rejected(self):
        result = self.run_installer("system", "--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("separate review", result.stderr)

    def test_plan_apply_is_rejected(self):
        result = self.run_installer("plan", "--apply")
        self.assertNotEqual(result.returncode, 0)

    def test_apply_requires_root_argument(self):
        result = self.run_installer("environment", "--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--root is required", result.stderr)

    def test_missing_root_value_is_rejected(self):
        result = self.run_installer("environment", "--root")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--root needs a value", result.stderr)

    def test_unknown_option_is_rejected(self):
        result = self.run_installer("environment", "--download-models")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown option", result.stderr)

    def test_system_plan_never_executes_package_commands(self):
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                f"""
source {shlex.quote(str(INSTALLER))}
demo_load_helpers() {{ SETUP_PACKAGES=(git curl); }}
demo_main system
""",
            ],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sudo apt-get update", result.stdout)
        self.assertIn("--no-install-recommends git curl", result.stdout)
        self.assertFalse(self.log.exists())

    def test_wrong_account_is_rejected_before_installation(self):
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                f"""
source {shlex.quote(str(INSTALLER))}
id() {{ printf '%s\\n' administrator; }}
demo_main environment --root {shlex.quote(str(self.root))} --apply
""",
            ],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("as demo, without sudo", result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
