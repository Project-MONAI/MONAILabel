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

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import BackgroundTasks, HTTPException

from monailabel.config import settings
from monailabel.endpoints import infer, wsi_infer
from monailabel.endpoints.proxy import proxy_dicom
from monailabel.endpoints.user.auth import DEFAULT_USER, RBAC, User, can_annotate

from .context import BasicEndpointTestSuite

USER = User(username="u", roles=[settings.MONAI_LABEL_AUTH_ROLE_USER])
ANNOTATOR = User(username="a", roles=[settings.MONAI_LABEL_AUTH_ROLE_USER, settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR])
ADMIN = User(username="r", roles=[settings.MONAI_LABEL_AUTH_ROLE_ADMIN])


class RawData:
    def read(self):
        return b"xyz"


def upstream(*args, **kwargs):
    return SimpleNamespace(content=b"xyz", raw=RawData(), status_code=200, headers={})


class TestRolesForWrites(unittest.TestCase):
    """With authentication enabled, things that write need the role of the documented write route."""

    def setUp(self):
        patcher = patch.object(settings, "MONAI_LABEL_AUTH_ENABLE", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_can_annotate(self):
        self.assertFalse(can_annotate(USER))
        self.assertFalse(can_annotate(ADMIN))  # roles are matched exactly, as for PUT /datastore/label
        self.assertTrue(can_annotate(ANNOTATOR))
        with patch.object(settings, "MONAI_LABEL_AUTH_ENABLE", False):
            self.assertTrue(can_annotate(USER))
            self.assertTrue(can_annotate(DEFAULT_USER))

    def _infer_request(self, module, call, **kwargs):
        instance = MagicMock()
        instance.info.return_value = {}
        instance.infer.return_value = {"params": {}}
        instance.infer_wsi.return_value = {"params": {}}
        with patch.object(module, "app_instance", return_value=instance):
            call(BackgroundTasks(), "model", image="img1", **kwargs)
        return (instance.infer.call_args or instance.infer_wsi.call_args).args[0]

    def test_infer_can_save_a_label_only_with_the_annotator_role(self):
        params = json.dumps({"save_label": True, "label_tag": "final"})
        kwargs = {"params": params, "file": None, "label": None}
        request = self._infer_request(infer, infer.run_inference, can_save_label=False, **kwargs)
        self.assertIs(request["save_label"], False)
        request = self._infer_request(infer, infer.run_inference, can_save_label=True, **kwargs)
        self.assertIs(request["save_label"], True)

        wsi = wsi_infer.WSIInput(params={"save_label": True, "label_tag": "final"})
        request = self._infer_request(wsi_infer, wsi_infer.run_wsi_inference, wsi=wsi, can_save_label=False)
        self.assertIs(request["save_label"], False)
        request = self._infer_request(wsi_infer, wsi_infer.run_wsi_inference, wsi=wsi, can_save_label=True)
        self.assertIs(request["save_label"], True)

    def _post(self, user, headers=None):
        async def body():
            return b"dicom"

        request = SimpleNamespace(method="POST", headers=headers or {}, body=body)
        with patch("requests.post", side_effect=upstream) as post:
            asyncio.run(proxy_dicom(request, "stow", "studies", user))
        return post

    def test_proxy_post_needs_the_annotator_role(self):
        with self.assertRaises(HTTPException) as cm:
            self._post(USER)
        self.assertEqual(cm.exception.status_code, 403)
        self.assertEqual(self._post(ANNOTATOR).call_count, 1)

        with patch("requests.get", side_effect=upstream) as get:
            asyncio.run(proxy_dicom(SimpleNamespace(method="GET", headers={}), "qido", "studies", USER))
        self.assertEqual(get.call_count, 1)

    def test_proxy_forwards_only_content_headers(self):
        headers = {
            "host": "internal-admin.example",
            "authorization": "Bearer client-token",
            "cookie": "session=1",
            "transfer-encoding": "chunked",
            "content-length": "5",
            "x-forwarded-for": "10.0.0.1",
            "Content-Type": "multipart/related; boundary=x",
            "accept": "application/dicom+json",
        }
        sent = self._post(ANNOTATOR, headers).call_args.kwargs["headers"]
        self.assertEqual(
            {k.lower(): v for k, v in sent.items()},
            {"content-type": "multipart/related; boundary=x", "accept": "application/dicom+json"},
        )

    def test_reload_needs_the_admin_role(self):
        from monailabel.app import app

        route = [r for r in app.routes if getattr(r, "path", "") == "/reload"][0]
        dependencies = [d.call for d in route.dependant.dependencies]
        self.assertEqual([d.roles for d in dependencies if isinstance(d, RBAC)], [settings.MONAI_LABEL_AUTH_ROLE_ADMIN])

    def test_every_route_with_an_effect_has_a_role(self):
        from monailabel.app import app

        public = {"/", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc", "/favicon.ico", "/static"}
        public |= {"/auth/", "/auth/token", "/auth/token/valid"}
        missing = []
        for route in app.routes:
            path = getattr(route, "path", "")
            dependant = getattr(route, "dependant", None)
            if path in public or dependant is None:
                continue
            if not [d for d in dependant.dependencies if isinstance(d.call, RBAC)]:
                missing.append(path)
        self.assertEqual(missing, [])


class TestDatastoreUploads(BasicEndpointTestSuite):
    def _put_image(self, filename, content=b"data", image=None):
        url = "/datastore/image" + (f"?image={image}" if image else "")
        return self.client.put(url, files={"file": (filename, content)})

    def test_image_name_and_type(self):
        before = self.client.get("/datastore/?output=all").json()
        for filename, image in (
            ("sitecustomize.py", None),
            ("x.py", "json"),
            (".env", None),
            (".hidden.nii.gz", None),
            ("x.nii.gz", ".hidden"),
            ("x.nii.gz", "a:b"),
            ("x.sh", None),
            ("noextension", "labels"),
        ):
            res = self._put_image(filename, image=image)
            self.assertEqual(res.status_code, 400, f"{filename} / {image}")
        self.assertEqual(self.client.get("/datastore/?output=all").json(), before)

    def test_detached_image_headers_are_rejected(self):
        header = b"NRRD0004\ntype: uint8\ndimension: 1\nsizes: 4\nencoding: raw\ndata file: /etc/hostname\n\n"
        res = self._put_image("detached_upload_test.nrrd", header)
        self.assertEqual(res.status_code, 400)
        self.assertIn("detached", res.json()["detail"])

        res = self.client.put("/datastore/label?image=spleen_3", files={"label": ("l.nrrd", header)})
        self.assertEqual(res.status_code, 400)

        res = self.client.post("/infer/deepedit", files={"file": ("i.nrrd", header)})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
