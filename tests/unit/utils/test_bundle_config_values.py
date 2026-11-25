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
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.utils.others.generic import UnsafeConfigValueError, check_config_value, is_plain_filename


class TestCheckConfigValue(unittest.TestCase):
    def test_plain_data_is_returned_unchanged(self):
        for value in (
            None,
            True,
            3,
            0.5,
            "no_channel",
            "cuda:0",
            "",
            " leading space",
            "costs 5$ or 10%",
            "a@b.com",
            [1, 2, 3],
            (96, 96, 96),
            ["a", ["b", {"c": 1.5}]],
            {"roi": [96, 96, 96], "mode": "gaussian", "nested": {"x": None}},
            torch.device("cpu"),  # not JSON: can only come from the server, never from a client
        ):
            self.assertIs(check_config_value(value, "k"), value)

    def test_config_syntax_is_rejected(self):
        for value in (
            "$__import__('os').system('id')",
            "  $1+1",
            "\t$x",
            "$",
            "@network_def",
            "@::probe",
            "%other_config.json#key",
            "%/etc/passwd",
            ["ok", "$1"],
            ("ok", ["ok", "@x"]),
            {"a": "$1"},
            {"a": {"b": ["%x"]}},
            {"_target_": "os.system", "_args_": ["id"]},
            {"x": {"_target_": "os.system"}},
            [{"_target_": "os.system"}],
            {"_requires_": "$1"},
            {"+x": 1},
            {"$x": 1},
            {"@x": 1},
            {" _target_": "os.system"},
        ):
            with self.assertRaises(UnsafeConfigValueError, msg=repr(value)):
                check_config_value(value, "k")

    def test_error_is_a_value_error_and_does_not_echo_the_value(self):
        with self.assertRaises(ValueError) as cm:
            check_config_value("$__import__('os').system('secret-command')", "probe")
        self.assertIn("probe", str(cm.exception))
        self.assertNotIn("secret-command", str(cm.exception))

    def test_is_plain_filename(self):
        for name in ("model.pt", "model_epoch=3.pt", "a b.pt", ".hidden.pt", "m..pt"):
            self.assertTrue(is_plain_filename(name), name)
        for name in (
            "",
            ".",
            "..",
            "../model.pt",
            "a/model.pt",
            "/abs/model.pt",
            "..\\model.pt",
            "a\\b.pt",
            "a\x00b",
            None,
            1,
        ):
            self.assertFalse(is_plain_filename(name), repr(name))


class BundleFixture(unittest.TestCase):
    """A real (tiny) MONAI bundle on disk: its config is parsed by the real MONAI ConfigParser."""

    def setUp(self):
        self.base = os.path.realpath(tempfile.mkdtemp(prefix="monailabel_bundle_values_"))
        self.bundle = os.path.join(self.base, "bundle")
        os.makedirs(os.path.join(self.bundle, "configs"))
        os.makedirs(os.path.join(self.bundle, "models"))
        self.marker = os.path.join(self.base, "marker.txt")

        self._write_json(
            "metadata.json",
            {
                "version": "0.0.1",
                "description": "test bundle",
                "network_data_format": {
                    "inputs": {"image": {"spatial_shape": [8, 8, 8]}},
                    "outputs": {"pred": {"channel_def": {"0": "background", "1": "organ"}}},
                },
            },
        )
        self._write_json(
            "inference.json",
            {
                "device": "$torch.device('cpu')",
                "network_def": {"_target_": "torch.nn.Identity"},
                "displayable_configs": {"probe": "no_channel", "roi_size": [8, 8, 8]},
                "preprocessing": [
                    {"_target_": "LoadImaged", "keys": ["image"]},
                    {
                        "_target_": "EnsureChannelFirstd",
                        "keys": ["image"],
                        "channel_dim": "@displayable_configs#probe",
                    },
                ],
                "postprocessing": [],
            },
        )
        self._write_json(
            "train.json",
            {
                "device": "$torch.device('cpu')",
                "network_def": {"_target_": "torch.nn.Identity"},
                "displayable_configs": {"lr": 0.001},
            },
        )
        torch.save({}, os.path.join(self.bundle, "models", "model.pt"))

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _write_json(self, name, content):
        with open(os.path.join(self.bundle, "configs", name), "w") as fp:
            json.dump(content, fp)

    def expression_writing_marker(self, result="no_channel"):
        return f'$(__import__("pathlib").Path({self.marker!r}).write_text("executed") and {result!r})'


class TestBundleInferRequestValues(BundleFixture):
    def _task(self):
        task = BundleInferTask(self.bundle, {})
        self.assertTrue(task.is_valid())
        return task

    def test_plain_option_values_are_applied(self):
        task = self._task()
        pre = task.pre_transforms({"probe": "no_channel", "roi_size": [4, 4, 4], "device": "cpu"})
        self.assertEqual(len(pre), 2)
        self.assertEqual(task.bundle_config["displayable_configs"]["roi_size"], [4, 4, 4])
        self.assertFalse(os.path.exists(self.marker))

    def test_displayable_config_expression_is_not_evaluated(self):
        task = self._task()
        for value in (
            self.expression_writing_marker(),
            "%/etc/hostname",
            {"_target_": "pathlib.Path.write_text", "_args_": [self.marker, "executed"]},
            ["ok", self.expression_writing_marker()],
        ):
            with self.assertRaises(UnsafeConfigValueError):
                task.pre_transforms({"probe": value, "roi_size": [8, 8, 8]})
            self.assertFalse(os.path.exists(self.marker))
        # the rejected requests left the options untouched
        self.assertEqual(task.bundle_config["displayable_configs"]["probe"], "no_channel")

    def test_device_expression_is_not_evaluated(self):
        task = self._task()
        with self.assertRaises(UnsafeConfigValueError):
            task.pre_transforms(
                {"probe": "no_channel", "roi_size": [8, 8, 8], "device": self.expression_writing_marker("cpu")}
            )
        for call in (task.inferer, task.post_transforms):
            with self.assertRaises(UnsafeConfigValueError):
                call({"device": self.expression_writing_marker("cpu")})
        self.assertFalse(os.path.exists(self.marker))

    def test_rejected_device_does_not_leak_into_sys_path(self):
        task = self._task()
        before = list(sys.path)
        for call in (task.pre_transforms, task.inferer, task.post_transforms, task.detector):
            with self.assertRaises(UnsafeConfigValueError):
                call({"probe": "no_channel", "roi_size": [8, 8, 8], "device": "$1"})
        self.assertEqual(sys.path, before)

    def test_model_filename_must_be_a_file_in_the_models_folder(self):
        task = self._task()
        default = task.get_path()
        other = os.path.join(self.bundle, "models", "other.pt")
        torch.save({}, other)
        torch.save({}, os.path.join(self.base, "outside.pt"))

        def loaded(model_filename):
            with patch("monailabel.tasks.infer.basic_infer.torch.load", return_value={}) as load:
                task._networks.clear()
                task._get_network("cpu", {"model_filename": model_filename})
            return [os.path.realpath(c.args[0]) for c in load.call_args_list]

        self.assertEqual(loaded("other.pt"), [os.path.realpath(other)])
        for name in ("../../outside.pt", "../outside.pt", os.path.join(self.base, "outside.pt")):
            self.assertEqual(loaded(name), [os.path.realpath(default)], name)


class TestBundleTrainRequestValues(BundleFixture):
    def _run(self, **request):
        task = BundleTrainTask(self.bundle, {})
        self.assertTrue(task.is_valid())
        datastore = MagicMock()
        datastore.datalist.return_value = [{"image": f"{i}.nii.gz", "label": f"{i}_label.nii.gz"} for i in range(5)]
        base = {"model": "bundle", "val_split": 0.2, "multi_gpu": False, "device": "cpu", "lr": 0.001}
        base.update(request)
        with patch.object(task, "run_single_gpu") as run:
            task(base, datastore)
        return run

    def test_plain_request_values_reach_the_bundle(self):
        run = self._run(max_epochs=2, lr=0.5, tracking="None")
        overrides = run.call_args.args[1]
        self.assertEqual(overrides["displayable_configs"], {"lr": 0.5})
        self.assertEqual(overrides["train#trainer#max_epochs"], 2)

    def test_expressions_in_request_values_are_rejected(self):
        expr = self.expression_writing_marker()
        for key, value in (
            ("max_epochs", expr),
            ("lr", expr),
            ("lr", {"_target_": "os.system", "_args_": ["id"]}),
            ("device", expr),
            ("tracking_uri", expr),
            ("tracking_experiment_name", expr),
            ("tracking_run_name", expr),
            ("run_id", expr),
        ):
            request = {"tracking": "mlflow", key: value}
            with self.assertRaises(UnsafeConfigValueError, msg=key):
                self._run(**request)
            self.assertFalse(os.path.exists(self.marker), key)

    def test_model_filename_must_be_a_file_in_the_models_folder(self):
        for name in ("../../outside.pt", "/etc/passwd", "a/b.pt"):
            with self.assertRaises(ValueError, msg=name):
                self._run(model_filename=name)
        run = self._run(model_filename="model.pt")
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
