"""Account-scoped model listing. Never probes models with inference requests."""

import json
import time
from dataclasses import dataclass
from typing import cast

from httpx import Client, HTTPError, Timeout

from monailabel.core.errors import DomainError
from monailabel.providers.catalog.compatibility import compatible
from monailabel.providers.catalog.services import HostedService
from monailabel.providers.vision import VisionProvider


@dataclass(frozen=True)
class CatalogModel:
    id: str
    name: str
    provider: VisionProvider
    url: str


@dataclass(frozen=True)
class Catalog:
    models: list[CatalogModel]
    excluded: int


def _page(http: Client, url: str, params: dict[str, str]) -> dict[str, object]:
    try:
        with http.stream("GET", url, params=params) as response:
            if response.status_code in {401, 403}:
                raise DomainError(
                    "The provider rejected this API key or its model-list permission."
                )
            if response.status_code == 429:
                raise DomainError("The provider's model list is busy. Try again shortly.")
            if response.status_code != 200:
                raise DomainError(
                    "The provider could not list models. Try again later.", status=502
                )
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > 4_000_000:
                    raise DomainError(
                        "The provider returned an oversized model catalog.", status=502
                    )
            result = json.loads(content)
    except (HTTPError, ValueError):
        # Never relay upstream errors, request headers or provider response bodies.
        raise DomainError(
            "Could not read the provider's model list. Try again later.", status=502
        ) from None
    if not isinstance(result, dict):
        raise DomainError("The provider returned an invalid model catalog.", status=502)
    return cast(dict[str, object], result)


def discover(service: HostedService, key: str) -> Catalog:
    headers = {"Authorization": f"Bearer {key}"}
    params: dict[str, str] = {}
    if service.id == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        params = {"limit": "1000"}
    elif service.id == "gemini":
        headers = {"x-goog-api-key": key}
        params = {"pageSize": "1000"}
    elif service.id == "nvidia":
        params = {"include_metadata": "true"}
    models: dict[str, CatalogModel] = {}
    identifiers: set[str] = set()
    unsupported: set[str] = set()
    cursors: set[str] = set()
    deadline = time.monotonic() + 45
    with Client(headers=headers, timeout=Timeout(15, connect=5), follow_redirects=False) as http:
        for _ in range(20):
            if time.monotonic() > deadline:
                break
            page = _page(http, service.catalog_url, params)
            rows = page.get("models" if service.id == "gemini" else "data")
            if not isinstance(rows, list) or len(identifiers) + len(rows) > 10_000:
                raise DomainError("The provider returned an invalid model catalog.", status=502)
            for row in rows:
                if not isinstance(row, dict):
                    raise DomainError("The provider returned an invalid model entry.", status=502)
                identifier = row.get("name" if service.id == "gemini" else "id")
                if not isinstance(identifier, str) or not identifier or len(identifier) > 300:
                    raise DomainError("The provider returned an invalid model ID.", status=502)
                if service.id == "gemini":
                    identifier = identifier.removeprefix("models/")
                    if not identifier:
                        raise DomainError("The provider returned an invalid model ID.", status=502)
                identifiers.add(identifier)
                if not compatible(service.id, identifier, row):
                    unsupported.add(identifier)
                    models.pop(identifier, None)
                elif identifier not in unsupported:
                    name = row.get("display_name", row.get("displayName", identifier))
                    responses = service.responses_url and row.get("mode") == "responses"
                    models[identifier] = CatalogModel(
                        identifier,
                        name if isinstance(name, str) and name.strip() else identifier,
                        "openai-polygons" if responses else service.provider,
                        str(service.responses_url) if responses else service.inference_url,
                    )
            cursor = (
                page.get("nextPageToken")
                if service.id == "gemini"
                else (page.get("last_id") if page.get("has_more") else None)
            )
            if not cursor and not page.get("has_more"):
                return Catalog(
                    list(models.values()),
                    len(identifiers) - len(models),
                )
            if not isinstance(cursor, str) or not cursor or cursor in cursors or len(cursor) > 2000:
                raise DomainError(
                    "The provider returned invalid model-list pagination.", status=502
                )
            cursors.add(cursor)
            params["pageToken" if service.id == "gemini" else "after_id"] = cursor
    raise DomainError("The provider's model list exceeded the discovery limit.", status=502)
