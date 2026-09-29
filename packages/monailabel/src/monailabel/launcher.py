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

"""Start the workspace by default, retaining the existing client subcommands."""

import sys

CLIENT_COMMANDS = {
    "demo",
    "seed",
    "projects",
    "login",
    "chat",
    "import",
    "request",
    "wait",
    "viewer",
}


def main() -> None:
    args = sys.argv[1:]
    if args and (
        args[0] in CLIENT_COMMANDS or args[0] in {"client", "--url"} or args[0].startswith("--url=")
    ):
        if args[0] == "client":
            del sys.argv[1]
        from monailabel.client.cli import main as client_main

        client_main()
    else:
        if args and args[0] == "server":
            del sys.argv[1]
        from monailabel.server.main import main as server_main

        server_main()


if __name__ == "__main__":
    main()
