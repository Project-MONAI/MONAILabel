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

"""Container health probe for HTTP and locally certified HTTPS."""

import os
import ssl
import urllib.error
import urllib.request

from monailabel.server.workspace import workspace_dir


def check() -> None:
    address = "127.0.0.1:" + os.environ.get("MONAILABEL_HEALTH_PORT", "8000")
    path = address + "/api/auth/status"
    try:
        with urllib.request.urlopen("http://" + path, timeout=2):
            return
    except (OSError, urllib.error.URLError):
        authority = workspace_dir() / ".tls" / "ca.crt"
        context = ssl.create_default_context(cafile=str(authority) if authority.exists() else None)
        with urllib.request.urlopen("https://" + path, context=context, timeout=2):
            return


if __name__ == "__main__":
    check()
