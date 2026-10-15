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

from types import SimpleNamespace

import pytest

from monailabel.server import main, network


@pytest.fixture
def addresses(monkeypatch):
    monkeypatch.delenv("MONAILABEL_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(network.socket, "gethostname", lambda: "workstation")
    monkeypatch.setattr(network.socket, "getfqdn", lambda: "workstation.example")
    monkeypatch.setattr(
        network.socket,
        "getaddrinfo",
        lambda *args: [(network.socket.AF_INET, 0, 0, "", ("127.0.1.1", 0))],
    )

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, address):
            assert address == ("192.0.2.1", 9)

        def getsockname(self):
            return "10.0.0.42", 12345

    monkeypatch.setattr(network.socket, "socket", lambda *args: Probe())


def test_defaults_allow_machine_and_lan_addresses(addresses):
    hosts = network.allowed_hosts()
    assert {
        "localhost",
        "127.0.0.1",
        "127.0.1.1",
        "workstation",
        "workstation.example",
        "10.0.0.42",
    } <= set(hosts)
    assert "*" not in hosts


def test_explicit_allowed_hosts_replace_discovery(addresses, monkeypatch):
    monkeypatch.setenv("MONAILABEL_ALLOWED_HOSTS", "localhost, proxy.example ,")
    assert network.allowed_hosts() == ["localhost", "proxy.example"]


def test_offline_startup_still_allows_local_access(addresses, monkeypatch):
    def unavailable(*args):
        raise OSError("No network route")

    monkeypatch.setattr(network.socket, "socket", unavailable)
    monkeypatch.setattr(network.socket, "getaddrinfo", unavailable)
    assert "localhost" in network.allowed_hosts()
    assert "127.0.0.1" in network.allowed_hosts()


@pytest.mark.parametrize(
    "args,host,port",
    [
        ([], "127.0.0.1", 8000),
        (["--host", "0.0.0.0", "--port", "8443"], "0.0.0.0", 8443),
        (["--host", "192.0.2.5", "--port", "8080"], "192.0.2.5", 8080),
    ],
)
def test_server_listen_default_and_override(tmp_path, monkeypatch, args, host, port):
    calls = []
    monkeypatch.setattr("sys.argv", ["monailabel-server", "--assistant", "local", *args])
    monkeypatch.setattr(main, "configure_workspace", lambda _: tmp_path)
    monkeypatch.setattr(
        main, "create_app", lambda *args, **kwargs: SimpleNamespace(state=SimpleNamespace())
    )
    monkeypatch.setattr(main.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    main.main()
    assert calls[0]["host"] == host
    assert calls[0]["port"] == port
    assert calls[0]["proxy_headers"] is False
    assert calls[0]["forwarded_allow_ips"] == ""
    assert calls[0]["ws"] == "wsproto"
    assert calls[0]["ws_max_size"] == 16 * 1024 * 1024
    assert calls[0]["ws_per_message_deflate"] is False


def test_proxy_trust_is_explicit_and_does_not_inherit_wildcard_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    monkeypatch.setenv("MONAILABEL_DEMO_HTTP_LIMITS", "1")
    monkeypatch.setattr(
        "sys.argv",
        ["monailabel-server", "--trusted-proxy", "127.0.0.1", "--trusted-proxy", "::1"],
    )
    monkeypatch.setattr(main, "configure_workspace", lambda _: tmp_path)
    monkeypatch.setattr(
        main, "create_app", lambda *args, **kwargs: SimpleNamespace(state=SimpleNamespace())
    )
    monkeypatch.setattr(main.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    main.main()
    assert calls[0]["proxy_headers"] is True
    assert calls[0]["forwarded_allow_ips"] == "127.0.0.1,::1"
    assert calls[0]["limit_concurrency"] == 64


@pytest.mark.parametrize("value", ["*", "0.0.0.0/0", "localhost", "127.0.0.1,::1", "fe80::1%eth0"])
def test_proxy_trust_rejects_broad_or_ambiguous_values(monkeypatch, value):
    monkeypatch.setattr("sys.argv", ["monailabel-server", "--trusted-proxy", value])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2


def test_network_host_access_still_requires_login(http, monkeypatch):
    # The live app's default policy contains this machine's LAN address.
    hostname = network.socket.gethostname()
    http.cookies.clear()
    response = http.get("/", headers={"host": f"{hostname}:8000"})
    assert response.status_code == 200
    assert http.get("/api/projects", headers={"host": f"{hostname}:8000"}).status_code == 401
    assert http.get("/", headers={"host": "unknown-host.example"}).status_code == 400
