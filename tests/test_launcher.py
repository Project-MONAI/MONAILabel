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

"""One entry command starts the server while preserving existing client commands."""

import sys

import pytest
from monailabel.launcher import main


@pytest.mark.parametrize("arguments", [[], ["--port", "8123"], ["server", "--host", "127.0.0.1"]])
def test_launcher_routes_to_server(monkeypatch, arguments):
    calls = []
    monkeypatch.setattr(sys, "argv", ["monailabel", *arguments])
    monkeypatch.setattr("monailabel.server.main.main", lambda: calls.append(sys.argv[1:]))
    main()
    assert calls == [arguments[1:] if arguments[:1] == ["server"] else arguments]


@pytest.mark.parametrize(
    "arguments",
    [
        ["projects"],
        ["--url", "http://localhost:8123", "projects"],
        ["--url=http://localhost:8123", "projects"],
        ["client", "--help"],
    ],
)
def test_launcher_preserves_client_commands(monkeypatch, arguments):
    calls = []
    monkeypatch.setattr(sys, "argv", ["monailabel", *arguments])
    monkeypatch.setattr("monailabel.client.cli.main", lambda: calls.append(sys.argv[1:]))
    main()
    assert calls == [arguments[1:] if arguments[:1] == ["client"] else arguments]
