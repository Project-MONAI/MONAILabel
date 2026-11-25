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

from monailabel.config import settings
from monailabel.endpoints import model
from monailabel.endpoints.user.auth import RBAC


def _roles(route):
    return [d.call.roles for d in route.dependant.dependencies if isinstance(d.call, RBAC)]


class TestModelEndpointRoles(unittest.TestCase):
    """Model files are loaded by the server, so every /model operation is restricted to administrators."""

    def test_all_model_routes_require_admin(self):
        routes = [r for r in model.router.routes if hasattr(r, "dependant")]
        self.assertEqual(
            {(m, r.path) for r in routes for m in r.methods},
            {
                ("GET", "/model/{model}"),
                ("GET", "/model/info/{model}"),
                ("PUT", "/model/{model}"),
                ("DELETE", "/model/{model}"),
            },
        )
        for route in routes:
            with self.subTest(methods=sorted(route.methods), path=route.path):
                self.assertEqual(_roles(route), [settings.MONAI_LABEL_AUTH_ROLE_ADMIN])


if __name__ == "__main__":
    unittest.main()
