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

import unittest

from monailabel.client import MONAILabelClient


# TODO:: Mock HTTP Server/Response
class TestClient(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_info(self):
        c = MONAILabelClient("http://127.0.0.1:8000/")
        c.get_server_url()
        c.set_server_url("http://127.0.0.1:8000")
        try:
            c.info()
        except:
            pass

    def test_next_sample(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.next_sample("random")
        except:
            pass

    def test_create_session(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.create_session("xyz", {"h": 1})
        except:
            pass

    def test_get_session(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.get_session("xyzd")
        except:
            pass

    def test_remove_session(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.remove_session("xyzd")
        except:
            pass

    def test_upload_image(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.upload_image("xyzd", "1234")
        except:
            pass

    def test_save_label(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.save_label("xyzd", "1234")
        except:
            pass

    def test_infer(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.infer("model", "1234", {"xyz": 1})
        except:
            pass

    def test_wsi_infer(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.wsi_infer("model", "1234", {"xyz": 1})
        except:
            pass

    def test_train_start(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.train_start("model", {"xyz": 1})
        except:
            pass

    def test_train_stop(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.train_stop()
        except:
            pass

    def test_train_status(self):
        try:
            c = MONAILabelClient("http://127.0.0.1:8000/")
            c.train_status()
        except:
            pass


if __name__ == "__main__":
    unittest.main()
