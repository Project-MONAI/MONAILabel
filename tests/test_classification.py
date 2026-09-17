import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from monailabel.core.errors import DomainError
from monailabel.core.models import (
    ClassificationObject,
    ClassificationProposal,
    ModelRecord,
    ObjectClassification,
)
from monailabel.providers.classification import RemoteClassifier

OBJECTS = [
    ClassificationObject(id="a", label_id=1, region={"x": 2, "y": 3, "width": 3, "height": 4}),
    ClassificationObject(id="b", label_id=1, region={"x": 10, "y": 8, "width": 2, "height": 3}),
]


@pytest.mark.parametrize("provider", ["openai-chat-polygons", "openai-polygons"])
def test_vision_classifies_exact_objects_with_custom_categories_and_abstention(
    monkeypatch, provider
):
    source = np.full((16, 16, 3), 0.5, dtype=np.float32)
    before = source.copy()
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        chat = provider == "openai-chat-polygons"
        content = body["messages"][1]["content"] if chat else body["input"][0]["content"]
        urls = [c["image_url"]["url"] if chat else c["image_url"] for c in content[1:]]
        images = [Image.open(io.BytesIO(base64.b64decode(u.split(",")[1]))) for u in urls]
        assert len(images) == 2 and images[0].size == images[1].size == (16, 16)
        assert not np.array_equal(images[0], images[1])
        schema = (
            body["response_format"]["json_schema"]["schema"]
            if chat
            else body["text"]["format"]["schema"]
        )
        assert schema["$defs"]["ClassifiedMarker"]["properties"]["category"]["anyOf"][0][
            "enum"
        ] == ["Type A", "Type B"]
        value = json.dumps(
            {
                "classifications": [
                    {"marker": 2, "category": None},
                    {"marker": 1, "category": "Type B"},
                ]
            }
        )
        payload = (
            {"choices": [{"finish_reason": "stop", "message": {"content": value}}]}
            if chat
            else {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": value}]}]
            }
        )
        return httpx.Response(200, json=payload)

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    model = ModelRecord(
        project_id="p",
        label_ids=[0, 1],
        name="Vision",
        provider=provider,
        config={"url": "https://unused.test", "model": "fixture"},
    )
    result = RemoteClassifier().classify(
        source, OBJECTS, ["Type A", "Type B"], "classify nuclei", model
    )
    assert {r.object_id: r.category for r in result} == {"a": "Type B", "b": None}
    assert len(calls) == 1
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize(
    "values",
    [
        [{"marker": 1, "category": "Type A"}],
        [{"marker": 1, "category": "Type A"}, {"marker": 1, "category": None}],
        [{"marker": 1, "category": "invented"}, {"marker": 2, "category": None}],
    ],
)
def test_invalid_classification_output_is_never_partially_accepted(monkeypatch, values):
    real = httpx.Client
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"classifications": values})},
            }
        ]
    }
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: real(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)), **kw
        ),
    )
    model = ModelRecord(
        project_id="p",
        label_ids=[0, 1],
        name="Vision",
        provider="openai-chat-polygons",
        config={"url": "https://unused.test", "model": "fixture"},
    )
    with pytest.raises(DomainError, match="exactly one valid category"):
        RemoteClassifier().classify(
            np.zeros((16, 16, 3), np.float32), OBJECTS, ["Type A", "Type B"], "", model
        )


@pytest.fixture
def classification_setup(client, http):
    project = client.post(
        "/api/projects",
        {
            "name": "Pathology",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Nuclei", "color": "#00ff00"},
            ],
        },
    )
    image = np.full((16, 16, 3), 128, np.uint8)
    content = io.BytesIO()
    Image.fromarray(image).save(content, "PNG")
    prefix = f"/api/projects/{project['id']}"
    asset = http.post(
        prefix + "/assets/upload",
        params={"name": "slide.png", "group_id": "slide"},
        content=content.getvalue(),
    ).json()
    model = client.post(
        prefix + "/models",
        {
            "name": "Vision",
            "provider": "openai-chat-polygons",
            "config": {"url": "https://unused.test", "model": "fixture"},
        },
    )
    request = {
        "message": "classify the annotated nuclei",
        "context": {
            "asset_id": asset["id"],
            "model_id": model["id"],
            "viewer_actions": ["classify_objects"],
            "classification": {
                "base_revision": 0,
                "categories": ["Type A", "Type B"],
                "objects": [o.model_dump() for o in OBJECTS],
            },
        },
    }
    return project, asset, request


def test_classification_job_crops_once_publishes_proposal_and_enforces_access(
    client, http, classification_setup
):
    project, asset, request = classification_setup
    calls = []

    class Classifier:
        def classify(self, image, objects, categories, prompt, model):
            calls.append(image.shape)
            assert (objects[0].region.x, objects[0].region.y) == (0, 0)
            assert (objects[1].region.x, objects[1].region.y) == (8, 5)
            return [
                ObjectClassification(object_id=o.id, category="Type A" if o.id == "a" else None)
                for o in objects
            ]

    http.app.state.services.models.classifiers["openai-chat-polygons"] = Classifier()
    reply = client.post(f"/api/projects/{project['id']}/assistant", request)
    result = client.wait(reply["job_id"])
    proposal_path = "/api/classification-proposals/" + result["classification_id"]
    proposal = client.get(proposal_path)
    assert proposal["results"] == [
        {"object_id": "a", "category": "Type A"},
        {"object_id": "b", "category": None},
    ]
    assert calls == [(8, 10, 3)]
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
    client.post("/api/auth/users", {"username": "outsider", "password": "test-password-1234"})
    client.post("/api/auth/login", {"username": "outsider", "password": "test-password-1234"})
    assert http.get(proposal_path).status_code == 403


def test_stale_or_invalid_classification_creates_no_proposal(client, http, classification_setup):
    project, _, request = classification_setup
    prefix = f"/api/projects/{project['id']}"
    request["context"]["classification"]["base_revision"] = 1
    assert http.post(prefix + "/assistant", json=request).status_code == 409
    assert client.get(prefix + "/jobs") == []
    request["context"]["classification"]["base_revision"] = 0

    class Classifier:
        def classify(self, *args):
            return [ObjectClassification(object_id="wrong", category="Type A")]

    http.app.state.services.models.classifiers["openai-chat-polygons"] = Classifier()
    reply = client.post(prefix + "/assistant", request)
    with pytest.raises(RuntimeError, match="invalid object categories"):
        client.wait(reply["job_id"])
    assert http.app.state.services.store.list(ClassificationProposal, project["id"]) == []
