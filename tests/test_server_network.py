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


@pytest.mark.parametrize("args,host", [([], "0.0.0.0"), (["--host", "127.0.0.1"], "127.0.0.1")])
def test_server_listen_default_and_override(tmp_path, monkeypatch, args, host):
    calls = []
    monkeypatch.setattr("sys.argv", ["monailabel-server", "--assistant", "local", *args])
    monkeypatch.setattr(main, "configure_workspace", lambda _: tmp_path)
    monkeypatch.setattr(
        main, "create_app", lambda *args, **kwargs: SimpleNamespace(state=SimpleNamespace())
    )
    monkeypatch.setattr(main.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    main.main()
    assert calls[0]["host"] == host
    assert calls[0]["port"] == 8000


def test_network_host_access_still_requires_login(http, monkeypatch):
    # The live app's default policy contains this machine's LAN address.
    hostname = network.socket.gethostname()
    http.cookies.clear()
    response = http.get("/", headers={"host": f"{hostname}:8000"})
    assert response.status_code == 200
    assert http.get("/api/projects", headers={"host": f"{hostname}:8000"}).status_code == 401
    assert http.get("/", headers={"host": "unknown-host.example"}).status_code == 400
