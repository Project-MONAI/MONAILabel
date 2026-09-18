"""Vision localization contracts, including abstentions and unsafe provider output."""

import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord
from monailabel.providers.tool_detection import RemoteToolDetector


@pytest.mark.parametrize("provider", ["openai-chat-polygons", "openai-polygons"])
@pytest.mark.parametrize(
    "status,box", [("found", [1.5, 2, 31, 19]), ("not_found", None), ("ambiguous", None)]
)
def test_detection_source_geometry_model_and_abstention(monkeypatch, provider, status, box):
    def respond(request):
        assert request.headers["Authorization"] == "Bearer test-only-token"
        body = json.loads(request.content)
        assert body["model"] == "requested-vision-model"
        chat = provider == "openai-chat-polygons"
        content = body["messages"][-1]["content"] if chat else body["input"][0]["content"]
        assert json.loads(content[0]["text"]) == {
            "width": 32,
            "height": 24,
            "tool": "Grasper",
            "request": "leftmost grasper",
        }
        image_url = content[1]["image_url"]["url"] if chat else content[1]["image_url"]
        assert Image.open(io.BytesIO(base64.b64decode(image_url.split(",")[1]))).size == (32, 24)
        schema = body["response_format"]["json_schema"] if chat else body["text"]["format"]
        assert schema["strict"] is True
        assert set(schema["schema"]["required"]) == {"status", "box"}
        if chat:
            assert body["reasoning_effort"] == "low"
            assert body["max_completion_tokens"] == 4096
        else:
            assert body["reasoning"] == {"effort": "low"}
            assert body["store"] is False
        text = json.dumps({"status": status, "box": box})
        payload = (
            {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
            if chat
            else {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            }
        )
        return httpx.Response(200, json=payload)

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(respond), **kw)
    )
    model = ModelRecord(
        project_id="project",
        name="Requested model",
        provider=provider,
        label_ids=[0],
        config={
            "url": "https://vision.test/v1",
            "model": "requested-vision-model",
            "credential_id": "saved",
            "reasoning_effort": "low",
        },
    )
    calls = []

    def credential(project_id, identifier):
        calls.append((project_id, identifier))
        return "test-only-token"

    result = RemoteToolDetector(credential).locate(
        np.zeros((24, 32, 3), np.float32), Label(id=1, name="Grasper"), "leftmost grasper", model
    )
    assert result.status == status and result.box == box
    assert calls == [("project", "saved")]


@pytest.mark.parametrize(
    "result",
    [
        {"status": "found", "box": None},
        {"status": "ambiguous", "box": [1, 2, 3, 4]},
        {"status": "found", "box": [-1, 2, 3, 4]},
        {"status": "found", "box": [1, 2, 33, 4]},
        {"status": "found", "box": [1, 2, 3, 25]},
        {"status": "found", "box": [3, 2, 1, 4]},
        {"status": "found", "box": [1, 2, 3]},
        {"status": "found", "box": [1, 2, float("nan"), 4]},
    ],
)
def test_detection_rejects_invalid_output(monkeypatch, result):
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}
    check_bad_response(monkeypatch, payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"status":"found","box":[1,2,3,4]}'},
                }
            ]
        },
        {"choices": [{"finish_reason": "stop", "message": {"refusal": "sensitive provider text"}}]},
        {"choices": []},
        {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "private arbitrary response"}}
            ]
        },
    ],
)
def test_detection_rejects_incomplete_and_refused_output(monkeypatch, payload):
    check_bad_response(monkeypatch, payload)


def check_bad_response(monkeypatch, payload):
    real = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: real(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)), **kw
        ),
    )
    model = ModelRecord(
        project_id="project",
        name="Sol",
        provider="openai-chat-polygons",
        label_ids=[0],
        config={"url": "https://vision.test/v1", "model": "sol"},
    )
    with pytest.raises(DomainError, match="No track was created") as error:
        RemoteToolDetector(lambda *_: "").locate(
            np.zeros((24, 32, 3), np.float32), Label(id=1, name="tool"), "", model
        )
    assert "private" not in str(error.value) and "sensitive" not in str(error.value)
