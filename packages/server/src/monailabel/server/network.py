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

"""Default HTTP hostnames for local and LAN workspace access."""

import os
import socket
from contextlib import suppress


def allowed_hosts() -> list[str]:
    configured = os.environ.get("MONAILABEL_ALLOWED_HOSTS")
    if configured is not None:
        return [host.strip() for host in configured.split(",") if host.strip()]

    names = {"localhost", "127.0.0.1", "[::1]", "testserver", socket.gethostname().lower()}
    names.add(socket.getfqdn().lower())
    for name in tuple(names - {"testserver", "[::1]"}):
        with suppress(OSError):
            names.update(
                str(address[4][0]) for address in socket.getaddrinfo(name, None, socket.AF_INET)
            )
    # A hostname can resolve only to loopback. UDP connect selects the default
    # interface without sending a packet to this documentation-only address.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            names.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(names)
