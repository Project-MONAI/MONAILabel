"""Small HTTP client; workflows use public endpoints, not server internals."""

import hashlib
import os
import time
from typing import Any, Self

import httpx
from platformdirs import user_config_path


class Client:
    def __init__(self, url: str = "http://127.0.0.1:8000", *, http: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.token_path = user_config_path("monailabel") / (
            hashlib.sha256(self.url.encode()).hexdigest() + ".token"
        )
        token = os.environ.get("MONAILABEL_TOKEN")
        if token is None and self.token_path.is_file():
            token = self.token_path.read_text().strip()
        self.http = http or httpx.Client(
            base_url=self.url,
            timeout=180,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        self._owns_http = http is None

    def login(self, username: str, password: str, *, setup: bool = False) -> None:
        self.http.headers.pop("Authorization", None)
        self.post(
            "/api/auth/setup" if setup else "/api/auth/login",
            {"username": username, "password": password},
        )
        token = self.post("/api/auth/token")["token"]
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            self.token_path.chmod(0o600)
            stream.write(token)
        self.http.headers["Authorization"] = f"Bearer {token}"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        if self._owns_http:
            self.http.close()

    def request(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        response = self.http.request(method, path, json=data)
        if response.is_error:
            try:
                message = response.json().get("detail", response.text)
            except ValueError:
                message = f"HTTP {response.status_code}"
            raise RuntimeError(str(message))
        return response.json()

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, data: dict[str, Any] | None = None) -> Any:
        return self.request("POST", path, data)

    def wait(self, job_id: str, timeout: float = 300) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        interval = 0.05
        while time.monotonic() < deadline:
            job = self.get(f"/api/jobs/{job_id}")
            if job["status"] == "succeeded":
                return dict(job["result"])
            if job["status"] in {"failed", "cancelled", "interrupted"}:
                raise RuntimeError(job.get("error") or f"Job {job['status']}")
            time.sleep(interval)
            interval = min(1.0, interval * 1.5)
        raise TimeoutError(f"Job {job_id} is still running; poll it again or cancel it.")
