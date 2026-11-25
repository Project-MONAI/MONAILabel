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
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from monailabel.client.client import MONAILabelUtils
from monailabel.datastore.cvat import CVATDatastore
from monailabel.datastore.dsa import DSADatastore, is_object_id
from monailabel.datastore.local import LocalDatastore
from monailabel.datastore.utils.dicom import _instance_file_id
from monailabel.datastore.xnat import XNATDatastore
from monailabel.interfaces.exception import ImageNotFoundException
from monailabel.scribbles.transforms import MakeLikelihoodFromScribblesGMMd, MakeLikelihoodFromScribblesHistogramd
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.transform.writer import Writer
from monailabel.utils.others.generic import (
    UnsafeConfigValueError,
    UnsafeImageFileError,
    UnsafeRequestError,
    check_image_file,
    int_in_range,
    is_plain_filename,
)

from .test_bundle_config_values import BundleFixture


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.base = os.path.realpath(tempfile.mkdtemp(prefix="monailabel_hardening_"))

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def write(self, name, content: bytes):
        path = os.path.join(self.base, name)
        with open(path, "wb") as fp:
            fp.write(content)
        return path


class TestImageFileHeaders(TempDirTest):
    """Image headers must not make the loader read the pixel data from another file on the server."""

    NRRD = b"NRRD0004\ntype: uint8\ndimension: 1\nsizes: 4\nencoding: raw\n"

    def test_self_contained_files_are_accepted(self):
        for name, content in (
            ("a.nrrd", self.NRRD + b"\n\x00\x01\x02\x03"),
            ("b.nrrd", self.NRRD.replace(b"\n", b"\r\n") + b"\r\n\x00\x01\x02\x03"),
            # "data file:" after the header (inside the pixel data) is not a header field
            ("c.nrrd", self.NRRD + b"\ndata file: /etc/passwd\n"),
            (
                "d.mha",
                b"ObjectType = Image\nNDims = 1\nDimSize = 4\nElementType = MET_UCHAR\nElementDataFile = LOCAL\n\x00",
            ),
            ("e.nii.gz", b"\x1f\x8b\x08\x00" + os.urandom(64)),
            ("f.png", b"\x89PNG\r\n\x1a\n" + os.urandom(64)),
        ):
            path = self.write(name, content)
            self.assertEqual(check_image_file(path), path)

    def test_values_that_are_not_files_are_returned(self):
        for value in (None, "", "spleen_10", self.base, ["a"], 3):
            self.assertIs(check_image_file(value), value)

    def test_detached_data_is_rejected(self):
        padding = b"# " + b"x" * 4096 + b"\n"
        for name, content in (
            ("a.nrrd", self.NRRD + b"data file: /etc/passwd\n\n"),
            ("b.nrrd", self.NRRD + b"datafile: /etc/passwd\n\n"),
            ("c.nrrd", self.NRRD + b"Data File: ../../secret.raw\n\n"),
            ("d.nrrd", self.NRRD + b"data file: LIST\n\n/etc/passwd\n"),
            ("e.nhdr", self.NRRD + b"data file: ./e.raw\n\n"),
            ("f.bin", self.NRRD + b"data file: /etc/passwd\n\n"),  # by content, whatever the file is called
            ("g.nrrd", self.NRRD + padding * 300 + b"data file: /etc/passwd\n\n"),  # header larger than the limit
            ("h.mha", b"ObjectType = Image\nNDims = 1\nDimSize = 4\nElementDataFile = /etc/passwd\n"),
            ("i.mhd", b"ObjectType = Image\nNDims = 1\nDimSize = 4\nElementDataFile = i.raw\n"),
            ("j.mhd", b"ObjectType = Image\nNDims = 1\nDimSize = 4\n"),
            ("k.h5", b"\x89HDF\r\n\x1a\n" + os.urandom(64)),
            ("l.nii", b"\x00" * 512 + b"\x89HDF\r\n\x1a\n" + os.urandom(64)),  # HDF5 with a user block
        ):
            with self.assertRaises(UnsafeImageFileError, msg=name):
                check_image_file(self.write(name, content))

    def test_error_can_be_mapped_to_bad_request(self):
        self.assertTrue(issubclass(UnsafeImageFileError, UnsafeRequestError))
        self.assertTrue(issubclass(UnsafeConfigValueError, UnsafeRequestError))
        self.assertTrue(issubclass(UnsafeRequestError, ValueError))


class TestRequestNumbersAndNames(unittest.TestCase):
    def test_int_in_range(self):
        self.assertEqual(int_in_range(20, 1, 64), 20)
        self.assertEqual(int_in_range(20.0, 1, 64), 20)
        for value in (
            "20",
            "20 -DX=1",
            "20; id",
            [20],
            {"a": 1},
            None,
            True,
            0,
            65,
            1.5,
            1e9,
            float("inf"),
            float("nan"),
        ):
            with self.assertRaises(UnsafeRequestError, msg=repr(value)):
                int_in_range(value, 1, 64, "num_mixtures")

    def test_scribbles_parameters_must_be_numbers(self):
        # the GMM mixture size becomes a compiler flag of MONAI's GMM extension
        with self.assertRaises(UnsafeRequestError):
            MakeLikelihoodFromScribblesGMMd(image="image", scribbles="label")(
                {"num_mixtures": "20 -include /etc/passwd"}
            )
        with self.assertRaises(UnsafeRequestError):
            MakeLikelihoodFromScribblesGMMd(image="image", scribbles="label")({"num_mixtures": 100000})
        with self.assertRaises(UnsafeRequestError):
            MakeLikelihoodFromScribblesHistogramd(image="image", scribbles="label")({"num_bins": "64"})

    def test_windows_drive_and_stream_names_are_not_plain(self):
        for name in ("D:model.pt", "model.pt:stream", "C:"):
            self.assertFalse(is_plain_filename(name), name)

    def test_writer_extension_and_dtype(self):
        writer = Writer(label="pred")
        for key, value in (
            ("result_extension", "/../../x.nii.gz"),
            ("result_extension", "..\\x"),
            ("result_extension", ".nii.gz/../x"),
            ("result_extension", "nii"),
            ("result_extension", ["x"]),
            ("result_dtype", "U64"),
            ("result_dtype", "S100000"),
            ("result_dtype", "object"),
            ("result_dtype", "i4,i4"),
            ("result_dtype", {"names": ["a"], "formats": ["i4"]}),
            ("result_dtype", "nonsense"),
        ):
            with self.assertRaises(UnsafeRequestError, msg=f"{key}={value!r}"):
                writer({key: value, "image_path": "x.nii.gz"})
        # valid values get past the check (and then fail on the missing prediction)
        for request in ({"result_extension": ".nrrd"}, {"result_extension": ".seg.nrrd"}, {"result_dtype": "uint8"}):
            request["image_path"] = "x.nii.gz"
            with self.assertRaises(KeyError):
                writer(request)


class TestBundleTrainHardening(BundleFixture):
    def setUp(self):
        super().setUp()
        self._write_json(
            "train.json",
            {
                "device": "$torch.device('cpu')",
                "network_def": {"_target_": "torch.nn.Identity"},
                "displayable_configs": {"lr": 0.001},
                "train": {"trainer": {"max_epochs": 1}, "dataset": {"data": []}, "handlers": []},
                "validate": {"dataset": {"data": []}},
            },
        )
        self._write_json("multi_gpu_train.json", {"device": "$torch.device('cpu')"})
        with open(os.path.join(self.bundle, "configs", "logging.conf"), "w") as fp:
            fp.write("")

    def _task(self):
        task = BundleTrainTask(self.bundle, {})
        self.assertTrue(task.is_valid())
        return task

    def _run(self, task=None, **request):
        task = task if task else self._task()
        datastore = MagicMock()
        datastore.datalist.return_value = [{"image": f"/d/{i}.nii.gz", "label": f"/d/{i}_l.nii.gz"} for i in range(5)]
        base = {"model": "bundle", "val_split": 0.2, "multi_gpu": False, "device": "cpu", "lr": 0.001}
        base.update(request)
        with patch.object(task, "run_single_gpu") as single, patch.object(task, "run_multi_gpu") as multi:
            task(base, datastore)
        return single, multi

    def test_tracking_is_mlflow_or_nothing(self):
        # MONAI loads any other string as a config file (tracking settings), i.e. as bundle config to evaluate
        settings_file = os.path.join(self.base, "tracking.json")
        with open(settings_file, "w") as fp:
            json.dump({"handlers_id": {}, "configs": {"x": self.expression_writing_marker()}}, fp)
        for value in (settings_file, "/etc/passwd", "a.json,b.yaml", "None", "", ["x.json"], {"configs": {}}):
            single, _ = self._run(tracking=value)
            self.assertNotIn("tracking", single.call_args.args[1], repr(value))
        for value in ("mlflow", "MLflow", ["mlflow", "None"]):
            single, _ = self._run(tracking=value)
            self.assertEqual(single.call_args.args[1]["tracking"], "mlflow")

    def test_run_id_is_a_config_id(self):
        for value in ("--config_file=/tmp/x.json", "--tracking={'a':1}", "run --x", "'run'", "", "run;id", ["run"], 1):
            with self.assertRaises(UnsafeConfigValueError, msg=repr(value)):
                self._run(run_id=value)
        single, _ = self._run(run_id="train#trainer")
        single.assert_called_once()

    def test_multi_gpu_command_line_has_no_client_strings(self):
        # python-fire parses the command line again: "--x=y" is an override, quoted text and {..} are evaluated
        client_values = {
            "tracking_uri": "'" + self.expression_writing_marker() + "'",
            "tracking_experiment_name": "--config_file=/tmp/x.json",
            "tracking_run_name": "{'_target_': 'os.system'}",
        }
        _, multi = self._run(force_multi_gpu=True, tracking="mlflow", gpus="0", **client_values)
        cmd = multi.call_args.args[1]
        self.assertEqual(cmd[cmd.index("--tracking") + 1], "mlflow")
        self.assertNotIn("--tracking_uri", cmd)
        for value in client_values.values():
            self.assertFalse([c for c in cmd if value in c], value)
        self.assertEqual(cmd[cmd.index("run") + 1], "run")
        # the values are in the exported config file instead, as plain strings
        with open(os.path.join(self.bundle, "configs", "monailabel_train.json")) as fp:
            exported = json.load(fp)
        self.assertEqual(exported["tracking_uri"], client_values["tracking_uri"])

    def test_detection_boxes_from_label_files_are_numbers(self):
        task = self._task()
        label = os.path.join(self.base, "label.json")
        datastore = MagicMock()
        datastore.datalist.return_value = [{"image": "/d/1.nii.gz", "label": label}]

        with open(label, "w") as fp:
            json.dump({"markups": [{"center": [1, 2, 3], "size": [4, 5, 6.5]}]}, fp)
        ds = task._fetch_datalist({"model": "lung_nodule_ct_detection"}, datastore)
        self.assertEqual(ds[0]["box"], [[1.0, 2.0, 3.0, 4.0, 5.0, 6.5]])

        for markup in (
            {"center": [self.expression_writing_marker(), 2, 3], "size": [4, 5, 6]},
            {"center": [{"_target_": "os.system", "_args_": ["id"]}], "size": []},
            {"center": "$1", "size": "+1"},
            {"center": [[1, 2, 3]], "size": [4, 5, 6]},
        ):
            with open(label, "w") as fp:
                json.dump({"markups": [markup]}, fp)
            with self.assertRaises((ValueError, TypeError), msg=repr(markup)):
                task._fetch_datalist({"model": "lung_nodule_ct_detection"}, datastore)
        self.assertFalse(os.path.exists(self.marker))

    def test_datalist_entries_are_checked(self):
        task = self._task()
        datastore = MagicMock()
        datastore.datalist.return_value = [{"image": "$1", "label": "/d/1.nii.gz"} for _ in range(5)]
        with patch.object(task, "run_single_gpu"), self.assertRaises(UnsafeConfigValueError):
            task({"model": "bundle", "val_split": 0.2, "multi_gpu": False, "device": "cpu", "lr": 0.001}, datastore)


class TestLocalDatastoreHardening(TempDirTest):
    def _datastore(self):
        self.store = os.path.join(self.base, "store")
        os.makedirs(self.store)
        with open(os.path.join(self.store, "img1.nii.gz"), "wb") as fp:
            fp.write(b"image")
        return LocalDatastore(self.store, extensions=("*.nii.gz",), auto_reload=False)

    def test_info_that_cannot_be_read_back_is_rejected(self):
        ds = self._datastore()
        deep: dict = {}
        node = deep
        for _ in range(300):
            node["a"] = {}
            node = node["a"]
        for info in ({"client_id": "\ud800"}, {"\udfff": 1}, {"a": ["x", {"b": "\ud83d"}]}, deep):
            with self.assertRaises(ValueError):
                ds.update_image_info("img1", info)
        ds.update_image_info("img1", {"client_id": "user-1", "nested": {"a": [1, 2, {"b": "ok €"}]}})

        # the datastore file is still readable by a new process (training, scoring ... start one)
        again = LocalDatastore(self.store, extensions=("*.nii.gz",), auto_reload=False)
        self.assertEqual(again.get_image_info("img1")["client_id"], "user-1")

    def test_image_id_can_not_be_a_folder_or_a_path(self):
        ds = self._datastore()
        os.makedirs(os.path.join(self.store, "labels", "final"), exist_ok=True)
        upload = self.write("upload", b"data")  # no extension
        for image_id in ("labels", ".", "..", "a/b", "../x", "a\\b", "C:x"):
            with self.assertRaises(ValueError, msg=image_id):
                ds.add_image(image_id, upload, {})
        labels = os.path.join(self.store, "labels")
        self.assertEqual([f for f in os.listdir(labels) if not os.path.isdir(os.path.join(labels, f))], [])
        self.assertEqual(ds.add_image("img2", self.write("img2.nii.gz", b"data"), {}), "img2")


class TestRemoteDatastoreHardening(TempDirTest):
    def _dsa(self, asset_store_path=""):
        ds = object.__new__(DSADatastore)
        ds.api_url = "http://dsa/api/v1"
        ds.folders = {"0123456789abcdef01234567"}
        ds.asset_store_path = asset_store_path
        ds.cache_path = os.path.join(self.base, "cache")
        os.makedirs(ds.cache_path)
        ds.gc = MagicMock()
        ds.gc.get.return_value = []
        return ds

    def test_dsa_ids_are_object_ids(self):
        self.assertTrue(is_object_id("621e94e2b6881a7a4bef5170"))
        for value in ("../user/me", "621e94e2b6881a7a4bef5170/../x", "621E94E2B6881A7A4BEF5170", "", None, 1):
            self.assertFalse(is_object_id(value), repr(value))

    def test_dsa_unknown_id_is_never_a_path_or_a_rest_path(self):
        ds = self._dsa()
        secret = self.write("secret.txt", b"secret")
        for image_id in (secret, "../../" + os.path.basename(secret), "../user/me", "token/current?x="):
            self.assertEqual(ds.get_image_uri(image_id), "")
            with self.assertRaises(ImageNotFoundException):
                ds.get_image_info(image_id)
            with self.assertRaises(ImageNotFoundException):
                ds.get_image(image_id, {})
            with self.assertRaises(ImageNotFoundException):
                ds.get_label(image_id, "final")
        ds.gc.getItem.assert_not_called()
        for call in ds.gc.get.call_args_list:  # only the name search, with the name as a query parameter
            self.assertEqual(call.args[0], "item")

    def test_dsa_item_name_and_file_path_from_the_server_stay_inside(self):
        item_id = "621e94e2b6881a7a4bef5170"
        ds = self._dsa()
        for name in ("../../etc/passwd", "/etc/passwd", "a/b.svs", ".."):
            ds.gc.getItem.return_value = {"_id": item_id, "name": name}
            with self.assertRaises(ValueError, msg=name):
                ds.get_image_uri(item_id)
            with self.assertRaises(ValueError, msg=name):
                ds.get_image(item_id, {})

        ds.gc.getItem.return_value = {"_id": item_id, "name": "slide.svs"}
        self.assertEqual(ds.get_image_uri(item_id), f"http://dsa/api/v1/item/{item_id}")
        with open(os.path.join(ds.cache_path, "slide.svs"), "wb") as fp:
            fp.write(b"x")
        self.assertEqual(ds.get_image_uri(item_id), os.path.join(ds.cache_path, "slide.svs"))

        assets = os.path.join(self.base, "assets")
        os.makedirs(assets)
        secret = self.write("secret.txt", b"secret")
        ds = self._dsa_with_assets(assets, item_id, {"path": "../secret.txt"})
        self.assertEqual(ds.get_image_uri(item_id), f"http://dsa/api/v1/item/{item_id}")
        ds = self._dsa_with_assets(assets, item_id, {"path": secret})
        self.assertEqual(ds.get_image_uri(item_id), f"http://dsa/api/v1/item/{item_id}")
        with open(os.path.join(assets, "ok.svs"), "wb") as fp:
            fp.write(b"x")
        ds = self._dsa_with_assets(assets, item_id, {"path": "ok.svs"})
        self.assertEqual(ds.get_image_uri(item_id), os.path.join(assets, "ok.svs"))

    def _dsa_with_assets(self, assets, item_id, resource):
        shutil.rmtree(os.path.join(self.base, "cache"), ignore_errors=True)
        ds = self._dsa(asset_store_path=assets)
        ds.gc.getItem.return_value = {"_id": item_id, "name": "slide.svs", "largeImage": {"fileId": item_id}}
        ds.gc.get.return_value = resource
        return ds

    def test_xnat_project_allow_list_applies_to_info_and_label_upload(self):
        ds = object.__new__(XNATDatastore)
        ds.projects = {"Allowed"}
        ds._request_get = MagicMock()
        ds._request_put_file = MagicMock()
        for call in (
            lambda: ds.get_image_info("Other/S/E/1"),
            lambda: ds.save_label("Other/S/E/1", "label.nii.gz", "final", {}),
        ):
            with self.assertRaises(ValueError):
                call()
        ds._request_get.assert_not_called()
        ds._request_put_file.assert_not_called()

    def test_dicom_instance_uid_is_a_file_name(self):
        self.assertEqual(_instance_file_id("1.2.840.113619.2.55"), "1.2.840.113619.2.55")
        for uid in ("../../x", "/abs/x", "a/b", "..", "a\\b"):
            with self.assertRaises(ValueError, msg=uid):
                _instance_file_id(uid)

    def test_cvat_urls_must_be_on_the_cvat_server(self):
        ds = object.__new__(CVATDatastore)
        ds.api_url = "http://cvat:8080"
        self.assertTrue(ds._same_server("http://cvat:8080/api/jobs?task_id=1"))
        for url in (
            "http://169.254.169.254/latest/meta-data",
            "http://cvat:8081/api/jobs",
            "https://cvat:8080/api/jobs",
            "http://cvat:8080@evil.example/x",
            "http://evil.example/cvat:8080",
            "",
            None,
        ):
            self.assertFalse(ds._same_server(url), repr(url))


class TestClientDownloads(TempDirTest):
    def test_result_file_name_from_the_server_stays_in_tmpdir(self):
        outside = os.path.join(self.base, "outside.txt")
        tmpdir = os.path.join(self.base, "tmp")
        os.makedirs(tmpdir)
        for name in (outside, "../outside.txt", "..\\outside.txt", "a/../../outside.txt", "/etc/cron.d/x", ".bashrc"):
            saved = MONAILabelUtils.save_result({name: b"data"}, tmpdir)
            self.assertEqual(os.path.dirname(os.path.realpath(saved)), tmpdir, name)
            self.assertFalse(os.path.basename(saved).startswith("."), name)
        self.assertFalse(os.path.exists(outside))
        self.assertEqual(os.listdir(self.base), ["tmp"])


if __name__ == "__main__":
    unittest.main()
