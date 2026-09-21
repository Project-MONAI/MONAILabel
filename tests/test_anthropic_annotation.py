"""Direct API keys and arbitrary Claude model IDs across annotation operations."""

import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from monailabel.core.errors import DomainError
from monailabel.core.models import ClassificationObject, Label, ModelRecord
from monailabel.providers.classification import RemoteClassifier
from monailabel.providers.remote import RemoteSegmenter
from monailabel.providers.tool_detection import RemoteToolDetector

LABELS = [Label(id=0, name="Background"), Label(id=1, name="Tool")]


@pytest.fixture
def endpoint(monkeypatch):
    calls = []
    responses = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=responses.pop(0))

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(respond), **kw)
    )
    return calls, responses


def registered(**config):
    return ModelRecord(
        project_id="project",
        name="My custom Claude",
        provider="anthropic-polygons",
        label_ids=[0, 1],
        config={
            "url": "https://api.anthropic.com/v1/messages",
            "model": "my-vision-model",
            **config,
        },
    )


def complete(value, **updates):
    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps(value)}],
        **updates,
    }


@pytest.mark.parametrize("authentication", ["environment", "saved"])
@pytest.mark.parametrize("operation", ["segment", "locate", "classify"])
def test_native_messages_geometry_credentials_and_custom_models(
    endpoint, monkeypatch, authentication, operation
):
    calls, responses = endpoint
    monkeypatch.setenv("CUSTOM_CLAUDE_KEY", "test-only-api-key")
    model = registered(
        **(
            {"token_env": "CUSTOM_CLAUDE_KEY"}
            if authentication == "environment"
            else {"credential_id": "saved-key"}
        )
    )
    resolved = []

    def credentials(project, identifier):
        resolved.append((project, identifier))
        return "test-only-api-key"

    image = np.zeros((12, 16, 3), np.float32)
    if operation == "segment":
        responses.append(
            complete(
                {
                    "polygons": [
                        {
                            "label_id": 1,
                            "points": [{"x": 2, "y": 3}, {"x": 6, "y": 3}, {"x": 2, "y": 8}],
                        }
                    ]
                }
            )
        )
        result = RemoteSegmenter("anthropic-polygons", credentials).predict(
            image, LABELS, "tool", model
        )
        assert result.mask.shape == (12, 16)
        assert result.mask[3, 2] == 1 and result.mask[0, 0] == 0
    elif operation == "locate":
        responses.append(complete({"status": "found", "box": [2, 3, 6, 8]}))
        found = RemoteToolDetector(credentials).locate(image, LABELS[1], "tool", model)
        assert found.box == [2, 3, 6, 8]
    else:
        responses.append(complete({"classifications": [{"marker": 1, "category": "Type A"}]}))
        objects = [
            ClassificationObject(
                id="nucleus", label_id=1, region={"x": 2, "y": 3, "width": 4, "height": 5}
            )
        ]
        classified = RemoteClassifier(credentials).classify(image, objects, ["Type A"], "", model)
        assert [(item.object_id, item.category) for item in classified] == [("nucleus", "Type A")]
    assert len(calls) == 1
    request = calls[0]
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "test-only-api-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "my-vision-model" and body["max_tokens"] == 4096
    assert "reasoning_effort" not in body and "response_format" not in body
    assert "test-only-api-key" not in str(body)
    assert body["system"]
    content = body["messages"][0]["content"]
    images = [block["source"] for block in content if block["type"] == "image"]
    assert len(images) == (2 if operation == "classify" else 1)
    for source in images:
        assert source["type"] == "base64" and source["media_type"] == "image/png"
        assert Image.open(io.BytesIO(base64.b64decode(source["data"]))).size == (16, 12)
    schema = body["output_config"]["format"]
    assert schema["type"] == "json_schema"
    assert schema["schema"]["additionalProperties"] is False
    # API schemas omit unsupported constraints, while local output validation retains them.
    encoded_schema = json.dumps(schema)
    assert '"minimum"' not in encoded_schema and '"maxItems"' not in encoded_schema
    if operation == "segment":
        point = schema["schema"]["$defs"]["Point"]["properties"]["x"]
        assert "maximum: 16" in point["description"]
    assert resolved == ([] if authentication == "environment" else [("project", "saved-key")])


@pytest.mark.parametrize(
    "response",
    [
        complete({"polygons": []}, stop_reason="max_tokens"),
        complete({"polygons": []}, stop_reason="refusal"),
        complete({"polygons": []}, stop_reason="tool_use"),
        complete({"polygons": []}, content=[{"type": "refusal", "refusal": "No"}]),
        complete({"polygons": []}, content=[]),
        complete({"polygons": []}, content=[None]),
        complete({"polygons": []}, content=[{"type": "text", "text": "not JSON"}]),
        complete(
            {
                "polygons": [
                    {
                        "label_id": 1,
                        "points": [{"x": 0, "y": 0}, {"x": 99, "y": 0}, {"x": 0, "y": 2}],
                    }
                ]
            }
        ),
        complete({"polygons": [{"label_id": 1, "points": [{"x": 0, "y": 0}]}]}),
    ],
)
def test_native_invalid_or_incomplete_output_never_yields_a_mask(endpoint, response):
    calls, responses = endpoint
    responses.append(response)
    with pytest.raises(DomainError) as error:
        RemoteSegmenter("anthropic-polygons").predict(
            np.zeros((12, 16, 3), np.float32), LABELS, "tool", registered()
        )
    assert error.value.code in {"provider_output_invalid", "provider_output_truncated"}
    assert len(calls) == 1


def test_native_missing_key_fails_before_request(endpoint, monkeypatch):
    calls, _ = endpoint
    monkeypatch.delenv("CUSTOM_CLAUDE_KEY", raising=False)
    with pytest.raises(DomainError, match="Set CUSTOM_CLAUDE_KEY"):
        RemoteSegmenter("anthropic-polygons").predict(
            np.zeros((12, 16, 3), np.float32),
            LABELS,
            "tool",
            registered(token_env="CUSTOM_CLAUDE_KEY"),
        )
    assert calls == []
