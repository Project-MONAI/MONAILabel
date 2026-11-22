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
import io
import os
import shutil
import tarfile
import tempfile
import time
import unittest

from monailabel.config import settings

from .context import BasicEndpointTestSuite

# session_id = ""
# session_info = {}


class EndPointSession(BasicEndpointTestSuite):
    def test_001_get_invalid_session(self):
        response = self.client.get("/session/xyz")
        assert response.status_code != 200

    def test_002_create_session(self):
        # global session_id, session_info
        files = [("files", ("spleen_3.nii.gz", open(os.path.join(self.studies, "spleen_3.nii.gz"), "rb")))]

        response = self.client.put("/session/?expiry=600", files=files)
        assert response.status_code == 200

        r = response.json()
        session_id = r["session_id"]
        session_info = r["session_info"]

        assert session_id
        assert session_info
        assert os.path.isdir(os.path.join(settings.MONAI_LABEL_SESSION_PATH, session_id))

        # def test_003_get_session(self):
        #     global session_id, session_info
        response = self.client.get(f"/session/{session_id}")
        assert response.status_code == 200

        # def test_004_update_session_ts(self):
        #     global session_id, session_info
        time.sleep(1)
        response = self.client.get(f"/session/{session_id}?update_ts=true")
        assert response.status_code == 200
        assert response.json()["last_access_ts"] > session_info["last_access_ts"]

        # def test_005_get_session_image(self):
        response = self.client.get(f"/session/{session_id}?image=true")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"

        # def test_006_remove_session(self):
        #     global session_id, session_info
        response = self.client.delete(f"/session/{session_id}")
        assert response.status_code == 200
        assert not os.path.exists(os.path.join(settings.MONAI_LABEL_SESSION_PATH, session_id))

    def test_007_create_session_multiple_images(self):
        files = [
            ("files", ("spleen_3.nii.gz", open(os.path.join(self.studies, "spleen_3.nii.gz"), "rb"))),
            ("files", ("spleen_9.nii.gz", open(os.path.join(self.studies, "spleen_9.nii.gz"), "rb"))),
        ]
        params = {"client_id": "xyz"}

        response = self.client.put("/session/", files=files, data=params)
        assert response.status_code == 200

        response = self.client.delete(f"/session/{response.json()['session_id']}")
        assert response.status_code == 200

    def test_008_create_session_rejects_archive_member_outside_extract_dir(self):
        # keep the extraction directory (and the path the archive tries to reach) under a scratch directory
        base = tempfile.mkdtemp(prefix="monailabel_session_confine_")
        nested = os.path.join(base, "a", "b")
        os.makedirs(nested)

        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tf:
            info = tarfile.TarInfo("../../pwned.txt")
            info.size = 5
            tf.addfile(info, io.BytesIO(b"pwned"))

        tempdir = tempfile.tempdir
        tempfile.tempdir = nested
        try:
            files = [("files", ("archive.tar", archive.getvalue()))]
            response = self.client.put("/session/?uncompress=true", files=files)
        finally:
            tempfile.tempdir = tempdir

        try:
            assert response.status_code == 400
            assert not os.path.exists(os.path.join(base, "a", "pwned.txt"))
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
