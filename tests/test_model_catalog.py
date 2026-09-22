"""Account discovery, conservative filtering, credential isolation and safe import."""

import httpx
import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import ModelRecord, Project
from monailabel.providers.catalog import SERVICES, discover
from monailabel.providers.catalog.compatibility import compatible
from monailabel.providers.catalog.discovery import Catalog, CatalogModel
from monailabel.providers.catalog.presets import HOSTED_PRESETS, resolve_presets
from monailabel.providers.catalog.selection import recent_models


def test_nvidia_latest_versions_keep_variants_and_distinct_hosting_routes():
    identifiers = [
        "azure/openai/gpt-4o",
        "azure/openai/gpt-5.4",
        "switchyard/openai/gpt-5.5",
        "azure/openai/gpt-5.6-sol",
        "aws/openai/gpt-5.6-sol",
        "switchyard/openai/gpt-5.6-sol",
        "azure/openai/gpt-5.6-luna",
        "azure/openai/gpt-5.6-terra",
        "azure/openai/gpt-6-astra",
        "azure/anthropic/claude-opus-4-7",
        "azure/anthropic/claude-opus-4-8",
        "azure/anthropic/claude-opus-5",
        "aws/anthropic/bedrock-claude-opus-5",
        "azure/anthropic/claude-sonnet-4-6",
        "azure/anthropic/claude-sonnet-5",
        "gcp/google/gemini-3.5-flash",
        "gcp/google/gemini-3.7-flash",
        "gcp/google/gemini-3.8-flash",
        "gcp/google/gemini-3.1-pro-preview",
    ]
    models = [
        CatalogModel(i, i, "openai-chat-polygons", SERVICES["nvidia"].inference_url)
        for i in identifiers
    ]
    filtered = recent_models(Catalog(models, 3))
    hidden = {
        "azure/openai/gpt-4o",
        "azure/openai/gpt-5.4",
        "switchyard/openai/gpt-5.5",
        "azure/anthropic/claude-opus-4-7",
        "gcp/google/gemini-3.5-flash",
    }
    assert {m.id for m in filtered.models} == set(identifiers) - hidden
    assert filtered.excluded == 3 + len(hidden)


def test_nvidia_version_filter_is_specific_to_import_picker(
    http, hosted_presets, catalog_http, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "direct-key")
    versions = ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]

    def handler(request):
        if request.url.host == "inference-api.nvidia.com":
            return httpx.Response(200, json={"data": [{"id": "gcp/google/" + m} for m in versions]})
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "models/" + m, "supportedGenerationMethods": ["generateContent"]}
                    for m in versions
                ]
            },
        )

    catalog_http(handler)
    project = http.post("/api/projects", json={"name": "Version policy"}).json()
    prefix = f"/api/projects/{project['id']}"
    catalog = http.post(prefix + "/model-catalog", json={"service": "nvidia"}).json()
    assert [m["id"] for m in catalog["models"]] == [
        "gcp/google/gemini-3.7-flash",
        "gcp/google/gemini-3.8-flash",
    ]
    preset = next(m for m in http.get(prefix + "/models").json() if m["preset"] == "gemini-flash")
    routes = http.post(
        prefix + f"/models/{preset['id']}/model-catalog", json={"service": "nvidia"}
    ).json()
    assert [m["id"] for m in routes["models"]] == ["gcp/google/gemini-3.5-flash"]
    direct = http.post(prefix + "/model-catalog", json={"service": "gemini"}).json()
    assert len(direct["models"]) == 3


def test_preset_exposes_all_available_routes_and_keeps_the_selected_route(
    http, hosted_presets, catalog_http
):
    routes = ["azure/openai/gpt-6-astra", "aws/openai/gpt-6-astra", "switchyard/openai/gpt-6-astra"]
    catalog_http(
        lambda request: httpx.Response(
            200, json={"data": [{"id": route, "mode": "responses"} for route in routes]}
        )
    )
    project = http.post("/api/projects", json={"name": "Route choice"}).json()
    prefix = f"/api/projects/{project['id']}"
    astra = next(m for m in http.get(prefix + "/models").json() if m["preset"] == "nvidia-astra")
    path = prefix + f"/models/{astra['id']}"
    catalog = http.post(path + "/model-catalog", json={"service": "nvidia"}).json()
    assert {m["id"] for m in catalog["models"]} == set(routes)
    changed = http.post(
        path + "/provider",
        json={
            "base_version": astra["version"],
            "connection": {
                "service": "nvidia",
                "model": routes[1],
            },
        },
    )
    assert changed.status_code == 200
    assert changed.json()["config"]["model"] == routes[1]
    assert changed.json()["provider"] == "openai-polygons"


@pytest.mark.parametrize("gateway", ["all", "partial", "forbidden", "missing"])
def test_presets_prefer_gateway_and_fall_back_per_model(catalog_http, monkeypatch, gateway):
    for service in SERVICES.values():
        monkeypatch.setenv(service.token_env, "disposable-key")
    if gateway == "missing":
        monkeypatch.delenv("NV_INFERENCE_API_KEY")
    calls = []

    def handler(request):
        calls.append(request.url.host)
        assert request.method == "GET"
        if request.url.host == "inference-api.nvidia.com":
            if gateway == "forbidden":
                return httpx.Response(403, text="credential-not-exposed")
            specs = HOSTED_PRESETS if gateway == "all" else HOSTED_PRESETS[:1]
            return httpx.Response(
                200, json={"data": [{"id": ("gateway/provider/" + s.direct_model)} for s in specs]}
            )
        if request.url.host == "api.openai.com":
            return httpx.Response(200, json={"data": [{"id": "gpt-6-astra"}]})
        if request.url.host == "api.anthropic.com":
            return httpx.Response(200, json={"data": [{"id": "claude-opus-5"}]})
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-3.5-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )

    catalog_http(handler)
    resolved = resolve_presets()
    assert len(resolved) == 3
    if gateway == "all":
        assert calls == ["inference-api.nvidia.com"]
    else:
        assert calls.count("api.openai.com") == (0 if gateway == "partial" else 1)
    for spec in HOSTED_PRESETS:
        selected = resolved[spec.key]
        via_gateway = gateway == "all" or (gateway == "partial" and spec.key == "nvidia-astra")
        assert selected.config["model"] == (
            ("gateway/provider/" + spec.direct_model) if via_gateway else spec.direct_model
        )
        assert (
            selected.config["token_env"]
            == SERVICES["nvidia" if via_gateway else spec.direct_service].token_env
        )
        assert (
            selected.provider == SERVICES["nvidia" if via_gateway else spec.direct_service].provider
        )
    if gateway != "all":
        assert resolved["gemini-flash"].config["max_tokens_field"] == "max_tokens"


def test_unavailable_presets_are_hidden_and_historical_connections_preserved(http, hosted_presets):
    project = http.post("/api/projects", json={"name": "Automatic"}).json()
    service = http.app.state.services
    before = service.models.available(project["id"])
    astra = next(m for m in before if m.preset == "nvidia-astra")
    selected = service.models.set_default(project["id"], astra.id, project["version"])
    hosted_presets.hosted = {}
    hosted_presets.ensure(project["id"])
    assert {m.provider for m in service.models.available(project["id"])} == {
        "vista3d",
        "sam2",
        "medsam2",
    }
    previous = service.store.get(ModelRecord, astra.id)
    assert previous.archived and previous.config == astra.config
    project = service.store.get(Project, project["id"])
    assert project.annotation_model_id is None and project.version > selected.version


@pytest.mark.parametrize("astra_available", [True, False])
def test_retired_sol_preserves_history_and_imports_and_migrates_defaults(
    http, hosted_presets, astra_available
):
    project = http.post("/api/projects", json={"name": "Existing Sol project"}).json()
    service = http.app.state.services
    sol = ModelRecord(
        project_id=project["id"],
        name="GPT-5.6 Sol",
        provider="openai-polygons",
        label_ids=[0],
        preset="nvidia-sol",
        connection_mode="manual",
        config={"url": "https://unused.test", "model": "gpt-5.6-sol"},
    )
    imported = sol.model_copy(update={"id": "imported-sol", "preset": None, "name": "My Sol"})
    with service.store.transaction() as session:
        session.insert(sol)
        session.insert(imported)
        current = session.get(Project, project["id"])
        session.update(
            current.model_copy(update={"annotation_model_id": sol.id, "defaults": {1: sol.id}})
        )
    if not astra_available:
        hosted_presets.hosted.pop("nvidia-astra")
    hosted_presets.ensure(project["id"])
    active = service.models.available(project["id"])
    assert not any(m.preset == "nvidia-sol" for m in active)
    assert service.store.get(ModelRecord, sol.id) == sol.model_copy(
        update={"archived": True, "version": sol.version + 1}
    )
    assert service.models.get(project["id"], imported.id) == imported
    expected = next((m.id for m in active if m.preset == "nvidia-astra"), None)
    updated = service.store.get(Project, project["id"])
    assert updated.annotation_model_id == expected
    assert updated.defaults == ({1: expected} if expected else {})
    hosted_presets.ensure(project["id"])
    assert service.models.available(project["id"]) == active
    assert service.store.get(Project, project["id"]) == updated


def test_preset_provider_change_is_scoped_named_and_preserves_history(
    http, hosted_presets, catalog_http, monkeypatch
):
    from monailabel.server.models.presets import Presets

    monkeypatch.setenv("OPENAI_API_KEY", "disposable-direct-key")
    catalog_http(
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": ("gateway/provider/" + s.direct_model)
                        if request.url.host == "inference-api.nvidia.com"
                        else s.direct_model
                    }
                    for s in HOSTED_PRESETS
                ]
                + [{"id": "gpt-5.6-sol"}]
            },
        )
    )
    p = http.post("/api/projects", json={"name": "One"}).json()
    other = http.post("/api/projects", json={"name": "Two"}).json()
    prefix = f"/api/projects/{p['id']}"
    service = http.app.state.services
    astra = next(m for m in service.models.available(p["id"]) if m.preset == "nvidia-astra")
    service.models.set_default(p["id"], astra.id, p["version"])
    path = prefix + f"/models/{astra.id}"
    options = http.get(path + "/model-services").json()
    assert [s["id"] for s in options] == ["nvidia", "openai"]
    assert http.post(path + "/model-catalog", json={"service": "anthropic"}).status_code == 422
    assert (
        http.patch(path, json={"name": "Renamed preset", "base_version": astra.version}).status_code
        == 422
    )
    body = {
        "base_version": astra.version,
        "connection": {"service": "openai", "model": "gpt-6-astra", "name": "Ignored name"},
    }
    assert (
        http.post(f"/api/projects/{other['id']}/models/{astra.id}/provider", json=body).status_code
        == 403
    )
    changed = http.post(path + "/provider", json=body)
    assert changed.status_code == 200, changed.text
    new = changed.json()
    assert new["id"] != astra.id and new["name"] == "GPT-6 Astra"
    assert new["provider"] == "openai-polygons" and new["connection_mode"] == "manual"
    assert new["config"]["token_env"] == "OPENAI_API_KEY"
    assert service.store.get(ModelRecord, astra.id).config == astra.config
    assert service.store.get(ModelRecord, astra.id).archived
    assert service.store.get(Project, p["id"]).annotation_model_id == new["id"]
    assert http.post(path + "/provider", json=body).status_code in {409, 422}
    # A fresh startup resolver keeps the user's explicit project selection.
    monkeypatch.setenv("MONAILABEL_PRELOAD_MODELS", "1")
    restarted = Presets(service.store, service.secrets.resolve)
    restarted.ensure(p["id"])
    assert service.models.get(p["id"], new["id"]).connection_mode == "manual"
    assert (
        next(
            m for m in service.models.available(other["id"]) if m.preset == "nvidia-astra"
        ).provider
        == "openai-chat-polygons"
    )
    restored = http.post(
        prefix + f"/models/{new['id']}/provider", json={"base_version": new["version"]}
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["provider"] == "openai-chat-polygons"
    assert restored.json()["connection_mode"] == "automatic"
    imported = http.post(
        prefix + "/model-import",
        json={"service": "openai", "model": "gpt-6-astra", "name": "My Astra"},
    )
    assert imported.status_code == 201
    assert http.get(prefix + f"/models/{imported.json()['id']}/model-services").status_code == 422
    assert (
        http.post(
            prefix + f"/models/{imported.json()['id']}/provider", json={"base_version": 0}
        ).status_code
        == 422
    )
    duplicate_name = http.post(
        prefix + "/model-import",
        json={"service": "openai", "model": "gpt-5.6-sol", "name": "My Astra"},
    )
    assert duplicate_name.status_code == 422 and "name" in duplicate_name.text


@pytest.mark.parametrize(
    "service,identifier,row,expected",
    [
        ("openai", "gpt-6-astra", {}, True),
        ("openai", "gpt-5.6-sol-2026-02-16", {}, True),
        ("openai", "gpt-4o-2024-05-13", {}, False),
        ("openai", "gpt-4o-2024-08-06", {}, True),
        ("openai", "gpt-5.6-sol-batch", {}, False),
        ("openai", "gpt-4o-mini-transcribe", {}, False),
        ("openai", "gpt-image-2", {}, False),
        ("openai", "gpt-999", {}, False),
        ("openai", "o3-mini", {}, False),
        ("anthropic", "claude-opus-5", {}, True),
        ("anthropic", "claude-sonnet-4-5-20250929", {}, True),
        ("anthropic", "claude-sonnet-4-5-audio", {}, False),
        (
            "anthropic",
            "claude-opus-5",
            {"capabilities": {"image_input": {"supported": False}}},
            False,
        ),
        (
            "anthropic",
            "new-model",
            {
                "capabilities": {
                    "image_input": {"supported": True},
                    "structured_outputs": {"supported": True},
                }
            },
            True,
        ),
        ("anthropic", "new-model", {"capabilities": {"image_input": {"supported": True}}}, False),
        ("gemini", "gemini-3.8-flash", {"supportedGenerationMethods": ["generateContent"]}, True),
        ("gemini", "gemini-3.8-flash", {"supportedGenerationMethods": ["embedContent"]}, False),
        (
            "gemini",
            "gemini-3.1-flash-image",
            {"supportedGenerationMethods": ["generateContent"]},
            False,
        ),
        ("gemini", "unknown", {"supportedGenerationMethods": ["generateContent"]}, False),
        ("nvidia", "switchyard/openai/gpt-6-astra", {"mode": "chat"}, True),
        ("nvidia", "azure/anthropic/claude-opus-5", {}, True),
        ("nvidia", "aws/anthropic/bedrock-claude-sonnet-4-5-v1", {}, True),
        ("nvidia", "gcp/google/gemini-3.8-flash", {}, True),
        ("nvidia", "azure/openai/gpt-6-astra", {"mode": "responses"}, True),
        ("nvidia", "azure/openai/gpt-6-astra", {"mode": {}}, False),
        ("nvidia", "azure/openai/gpt-6-astra", {"supports_vision": False}, False),
        (
            "nvidia",
            "azure/openai/gpt-6-astra",
            {"metadata": {"supports_response_schema": "true"}},
            False,
        ),
        ("nvidia", "nvidia/unknown/vision", {}, False),
        (
            "nvidia",
            "nvidia/unknown/vision",
            {
                "metadata": {
                    "supports_vision": True,
                    "supports_response_schema": True,
                }
            },
            True,
        ),
        ("nvidia", "nvidia/unknown/vision", {"supports_vision": True}, False),
        (
            "nvidia",
            "nvidia/unknown/embedding",
            {"mode": "embedding", "supports_vision": True, "supports_response_schema": True},
            False,
        ),
    ],
)
def test_annotation_compatibility(service, identifier, row, expected):
    assert compatible(service, identifier, row) is expected


@pytest.mark.parametrize(
    "service,header",
    [
        ("nvidia", "authorization"),
        ("openai", "authorization"),
        ("anthropic", "x-api-key"),
        ("gemini", "x-goog-api-key"),
    ],
)
def test_discovery_protocol_filtering_and_pagination(catalog_http, service, header):
    calls = []

    def handler(request):
        assert request.method == "GET"
        assert str(request.url).split("?")[0] == SERVICES[service].catalog_url
        assert request.headers[header] == (
            "Bearer secret-key" if header == "authorization" else "secret-key"
        )
        assert "secret-key" not in str(request.url)
        calls.append(request)
        if service == "gemini":
            if len(calls) == 1:
                assert request.url.params["pageSize"] == "1000"
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {
                                "name": "models/embedding",
                                "supportedGenerationMethods": ["embedContent"],
                            },
                        ],
                        "nextPageToken": "next",
                    },
                )
            assert request.url.params["pageToken"] == "next"
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-3.8-flash",
                            "displayName": "Gemini 3.8 Flash",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                    ]
                },
            )
        if service == "anthropic":
            assert request.headers["anthropic-version"] == "2023-06-01"
            if len(calls) == 1:
                return httpx.Response(
                    200,
                    json={
                        "data": [{"id": "unsupported"}],
                        "has_more": True,
                        "last_id": "unsupported",
                    },
                )
            assert request.url.params["after_id"] == "unsupported"
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "claude-opus-5", "display_name": "Claude Opus 5"}],
                    "has_more": False,
                },
            )
        identifier = "azure/openai/gpt-6-astra" if service == "nvidia" else "gpt-6-astra"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "embedding"},
                    {"id": identifier},
                    {"id": identifier},
                ]
            },
        )

    catalog_http(handler)
    result = discover(SERVICES[service], "secret-key")
    assert len(result.models) == 1 and result.excluded == 1
    assert len(calls) == (2 if service in {"anthropic", "gemini"} else 1)
    assert "secret-key" not in repr(result)


@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
def test_catalog_errors_are_safe_and_redirects_not_followed(catalog_http, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status, text="provider-body-secret", headers={"location": "https://untrusted.example"}
        )

    catalog_http(handler)
    with pytest.raises(DomainError) as error:
        discover(SERVICES["openai"], "key-secret")
    assert "secret" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"data": "bad"},
        {"data": [None]},
        {"data": [{"id": 5}]},
        {"data": [], "has_more": True},
        {"data": [], "has_more": True, "last_id": "repeated"},
    ],
)
def test_invalid_catalog_fails_without_partial_results(catalog_http, payload):
    catalog_http(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(DomainError):
        discover(SERVICES["anthropic"], "key")


def test_transport_failure_redacts_request(catalog_http):
    def handler(request):
        raise httpx.ReadTimeout("secret-key", request=request)

    catalog_http(handler)
    with pytest.raises(DomainError, match="Could not read") as error:
        discover(SERVICES["openai"], "secret-key")
    assert "secret-key" not in str(error.value)


def test_gateway_responses_mode_selects_the_matching_inference_adapter(
    http, catalog_http, monkeypatch
):
    monkeypatch.setenv("NV_INFERENCE_API_KEY", "disposable-key")
    rows = [{"id": "azure/openai/gpt-6-astra", "mode": "responses"}]
    catalog_http(lambda request: httpx.Response(200, json={"data": rows}))
    project = http.post("/api/projects", json={"name": "Responses routes"}).json()
    body = {"service": "nvidia", "model": rows[0]["id"], "name": "My Astra"}
    prefix = f"/api/projects/{project['id']}"
    catalog = http.post(prefix + "/model-catalog", json={"service": "nvidia"}).json()
    assert catalog["models"][0]["provider"] == "openai-polygons"
    imported = http.post(prefix + "/model-import", json=body)
    assert imported.status_code == 201
    assert imported.json()["provider"] == "openai-polygons"
    assert imported.json()["config"]["url"] == "https://inference-api.nvidia.com/v1/responses"
    assert "max_tokens_field" not in imported.json()["config"]
    resolved = resolve_presets(tuple(s for s in HOSTED_PRESETS if s.key == "nvidia-astra"))
    assert resolved["nvidia-astra"].provider == "openai-polygons"
    assert resolved["nvidia-astra"].config["url"] == imported.json()["config"]["url"]


def test_contradictory_gateway_deployments_are_excluded(catalog_http):
    identifier = "azure/openai/gpt-6-astra"
    rows = [{"id": identifier}, {"id": identifier, "supports_vision": False}]
    for data in (rows, list(reversed(rows))):
        catalog_http(lambda request, data=data: httpx.Response(200, json={"data": data}))
        catalog = discover(SERVICES["nvidia"], "disposable-key")
        assert catalog.models == [] and catalog.excluded == 1
    assert not compatible(
        "nvidia",
        identifier,
        {
            "supports_vision": False,
            "metadata": {"supports_vision": True, "supports_response_schema": True},
        },
    )


def test_manual_preset_unavailability_does_not_switch_accounts(
    http, hosted_presets, catalog_http, monkeypatch
):
    from monailabel.server.models.presets import Presets

    monkeypatch.setenv("OPENAI_API_KEY", "direct-key")
    monkeypatch.setenv("MONAILABEL_PRELOAD_MODELS", "1")
    rows = [{"id": "gpt-6-astra"}]
    catalog_http(lambda request: httpx.Response(200, json={"data": rows}))
    project = http.post("/api/projects", json={"name": "Manual availability"}).json()
    service = http.app.state.services
    astra = next(m for m in service.models.available(project["id"]) if m.preset == "nvidia-astra")
    response = http.post(
        f"/api/projects/{project['id']}/models/{astra.id}/provider",
        json={
            "base_version": astra.version,
            "connection": {"service": "openai", "model": "gpt-6-astra"},
        },
    )
    assert response.status_code == 200
    manual = response.json()
    rows[:] = [{"id": "switchyard/openai/gpt-6-astra"}]
    restarted = Presets(service.store, service.secrets.resolve)
    restarted.ensure(project["id"])
    assert not any(m.preset == "nvidia-astra" for m in service.models.available(project["id"]))
    assert service.store.get(ModelRecord, manual["id"]).config == manual["config"]
    rows[:] = [{"id": "gpt-6-astra"}]
    restarted = Presets(service.store, service.secrets.resolve)
    restarted.ensure(project["id"])
    restored = service.models.get(project["id"], manual["id"])
    assert restored.connection_mode == "manual" and restored.provider == "openai-polygons"


def test_catalog_import_keeps_key_reference_and_rechecks(http, catalog_http, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    project = http.post("/api/projects", json={"name": "Catalog"}).json()
    prefix = f"/api/projects/{project['id']}"
    requests = []
    rows = [{"id": "gpt-6-astra"}, {"id": "text-embedding-3-small"}]

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": rows})

    catalog_http(handler)
    services = http.get(prefix + "/model-services")
    assert services.status_code == 200
    assert next(s for s in services.json() if s["id"] == "openai")["configured"] is True
    assert "environment-key" not in services.text
    catalog = http.post(prefix + "/model-catalog", json={"service": "openai"}).json()
    assert [m["id"] for m in catalog["models"]] == ["gpt-6-astra"]
    assert catalog["excluded"] == 1
    imported = http.post(
        prefix + "/model-import", json={"service": "openai", "model": "gpt-6-astra"}
    )
    assert imported.status_code == 201
    model = imported.json()
    assert model["provider"] == "openai-polygons"
    assert model["config"] == {
        "url": SERVICES["openai"].inference_url,
        "model": "gpt-6-astra",
        "token_env": "OPENAI_API_KEY",
        "timeout": 180,
        "max_output_tokens": 4096,
    }
    again = http.post(prefix + "/model-import", json={"service": "openai", "model": "gpt-6-astra"})
    assert again.json()["id"] == model["id"]
    catalog = http.post(prefix + "/model-catalog", json={"service": "openai"}).json()
    assert catalog["models"][0]["existing_model_id"] == model["id"]
    assert all(r.headers["authorization"] == "Bearer environment-key" for r in requests)
    rows.clear()
    unavailable = http.post(
        prefix + "/model-import", json={"service": "openai", "model": "gpt-6-astra"}
    )
    assert unavailable.status_code == 422
    assert len(http.get(prefix + "/models").json()) == 1


def test_saved_keys_are_scoped_and_rotated(http, catalog_http, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    project = http.post("/api/projects", json={"name": "One"}).json()
    other = http.post("/api/projects", json={"name": "Two"}).json()
    prefix = f"/api/projects/{project['id']}"
    key = http.post(prefix + "/credentials", json={"name": "Gemini", "api_key": "first-key"}).json()
    calls = []

    def handler(request):
        calls.append(request.headers["x-goog-api-key"])
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-3.8-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )

    catalog_http(handler)
    missing = http.post(prefix + "/model-catalog", json={"service": "gemini"})
    assert missing.status_code == 422 and "GEMINI_API_KEY" in missing.text
    body = {"service": "gemini", "credential_id": key["id"]}
    assert http.post(f"/api/projects/{other['id']}/model-catalog", json=body).status_code == 403
    assert calls == []
    assert http.post(prefix + "/model-catalog", json=body).status_code == 200
    http.post(
        prefix + "/credentials",
        json={"name": "Gemini", "credential_id": key["id"], "api_key": "rotated-key"},
    )
    imported = http.post(
        prefix + "/model-import", json=body | {"model": "gemini-3.8-flash", "name": "My Gemini"}
    )
    assert imported.status_code == 201
    config = imported.json()["config"]
    assert config["credential_id"] == key["id"] and "token_env" not in config
    assert config["max_tokens_field"] == "max_tokens"
    assert calls == ["first-key", "rotated-key"]
    assert all(k not in imported.text for k in calls)


def test_model_discovery_requires_project_manager(http, client, seeded):
    pid = seeded[0]["project_id"]
    user = client.post("/api/auth/users", {"username": "reader", "password": "test-password-1234"})
    client.request(
        "PUT", f"/api/projects/{pid}/members", {"user_id": user["id"], "roles": ["annotator"]}
    )
    http.post("/api/auth/login", json={"username": "reader", "password": "test-password-1234"})
    assert http.get(f"/api/projects/{pid}/model-services").status_code == 403
    assert (
        http.post(f"/api/projects/{pid}/model-catalog", json={"service": "openai"}).status_code
        == 403
    )
    assert (
        http.post(
            f"/api/projects/{pid}/model-import", json={"service": "openai", "model": "gpt-6-astra"}
        ).status_code
        == 403
    )
