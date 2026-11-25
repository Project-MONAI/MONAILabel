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

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from monailabel.utils.async_tasks import utils as task_utils
from monailabel.utils.async_tasks.task import AsyncTask


class FakeApp:
    """Only the registered bundle trainer ``bundle_model`` has a bundle path."""

    def __init__(self, bundle_dir):
        self.bundle_dir = bundle_dir

    def info(self):
        return {"config": {}}

    def bundle_path(self, model):
        return self.bundle_dir if model == "bundle_model" else None


class TestBundlePathIsServerControlled(unittest.TestCase):
    """``bundle_path`` ends up on the PYTHONPATH of the task process, so clients must not be able to set it."""

    def setUp(self):
        self.base = os.path.realpath(tempfile.mkdtemp(prefix="monailabel_bundle_path_"))
        self.bundle_dir = os.path.join(self.base, "server_bundle")
        self.client_dir = os.path.join(self.base, "client_dir")
        os.makedirs(self.bundle_dir)
        os.makedirs(self.client_dir)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _submitted_request(self, method, request=None, params=None):
        with (
            patch("monailabel.utils.async_tasks.task.app_instance", return_value=FakeApp(self.bundle_dir)),
            patch("monailabel.utils.async_tasks.task.processes", return_value={}),
            patch("monailabel.utils.async_tasks.task.run_background_task") as run,
        ):
            AsyncTask.run(method, request=request, params=params)
        run.assert_called_once()
        return run.call_args.args[0]

    def test_client_params_cannot_set_bundle_path(self):
        for method, request in (
            ("train", {"model": "unknown_model"}),
            ("train", {"model": "unknown_model", "bundle_path": self.client_dir}),
            ("train", {}),
            ("scoring", {"method": "any"}),
            ("batch_infer", {"model": "unknown_model"}),
        ):
            with self.subTest(method=method, request=request):
                submitted = self._submitted_request(
                    method, request=dict(request), params={"bundle_path": self.client_dir, "max_epochs": 1}
                )
                self.assertNotIn("bundle_path", submitted)
                self.assertEqual(submitted["max_epochs"], 1)

    def test_client_bundle_path_cannot_replace_server_bundle_path(self):
        submitted = self._submitted_request(
            "train", request={"model": "bundle_model"}, params={"bundle_path": self.client_dir}
        )
        self.assertEqual(submitted["bundle_path"], self.bundle_dir)

    def test_registered_bundle_model_gets_server_bundle_path(self):
        submitted = self._submitted_request("train", request={"model": "bundle_model"}, params={"max_epochs": 2})
        self.assertEqual(submitted["bundle_path"], self.bundle_dir)
        self.assertEqual(submitted["max_epochs"], 2)

    def _run_task(self, request, pythonpath):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if pythonpath is not None:
            env["PYTHONPATH"] = pythonpath

        process = MagicMock()
        process.poll.return_value = 0
        process.returncode = 0
        task = {"id": "task-1", "status": "SUBMITTED", "request": request}
        with (
            patch.dict(task_utils.background_processes, {"train": {}}),
            patch.dict(os.environ, env, clear=True),
            patch("monailabel.utils.async_tasks.utils.subprocess.Popen", return_value=process) as popen,
        ):
            task_utils._task_func(task, "train")
        popen.assert_called_once()
        return popen.call_args.kwargs["env"].get("PYTHONPATH")

    def test_task_process_pythonpath(self):
        self.assertEqual(self._run_task({"model": "m"}, "/app"), "/app")
        self.assertIsNone(self._run_task({"model": "m"}, None))
        self.assertEqual(
            self._run_task({"model": "m", "bundle_path": self.bundle_dir}, "/app"),
            os.pathsep.join(["/app", self.bundle_dir]),
        )
        # no PYTHONPATH set (e.g. server not started through the CLI) and no empty leading entry
        self.assertEqual(self._run_task({"model": "m", "bundle_path": self.bundle_dir}, None), self.bundle_dir)
        self.assertEqual(self._run_task({"model": "m", "bundle_path": self.bundle_dir}, ""), self.bundle_dir)
        # a path that does not exist is not added
        missing = os.path.join(self.base, "missing")
        self.assertEqual(self._run_task({"model": "m", "bundle_path": missing}, "/app"), "/app")


if __name__ == "__main__":
    unittest.main()
