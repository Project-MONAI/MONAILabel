import base64
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord
from monailabel.providers.remote import RemoteSegmenter

LABELS = [
    Label(id=0, name="Background", color="#000000"),
    Label(id=1, name="Liver", color="#ff0000"),
]


def mock_endpoint(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


def model(provider, **config):
    return ModelRecord(
        project_id="p",
        name="Test",
        provider=provider,
        label_ids=[0, 1],
        config={"url": "https://provider.example/predict", **config},
    )


def test_http_mask_explicit_volume_contract_and_credential(monkeypatch):
    def handler(request):
        assert request.headers["authorization"] == "Bearer hidden-secret"
        data = json.loads(request.content)
        assert data["spatial_shape"] == [4, 3, 2]
        assert data["labels"][1]["name"] == "Liver"
        assert data["prompt"] == "liver boundary"
        return httpx.Response(200, json={"mask": np.ones((4, 3, 2), dtype=int).tolist()})

    mock_endpoint(monkeypatch, handler)
    provider = RemoteSegmenter("http-mask", lambda project, credential: "hidden-secret")
    predicted = provider.predict(
        np.zeros((4, 3, 2, 1), dtype=np.float32),
        LABELS,
        "liver boundary",
        model("http-mask", credential_id="key"),
    )
    assert predicted.mask.shape == (4, 3, 2)
    assert predicted.mask.dtype == np.uint8


@pytest.mark.parametrize(
    "payload", [{"mask": [[1.0]]}, {"mask": [[9]]}, {"mask": [[1, 0]]}, {"no_mask": []}]
)
def test_remote_output_contract_is_enforced(monkeypatch, payload):
    mock_endpoint(monkeypatch, lambda request: httpx.Response(200, json=payload))
    with pytest.raises(DomainError, match="mask contract"):
        RemoteSegmenter("http-mask").predict(
            np.zeros((1, 1, 1), dtype=np.float32), LABELS, "", model("http-mask")
        )


def test_remote_error_does_not_expose_provider_body(monkeypatch):
    mock_endpoint(monkeypatch, lambda request: httpx.Response(401, text="secret-body-api-key"))
    with pytest.raises(DomainError) as error:
        RemoteSegmenter("http-mask").predict(
            np.zeros((1, 1, 1), dtype=np.float32), LABELS, "", model("http-mask")
        )
    assert "secret" not in str(error.value)
    assert "401" in str(error.value)


def test_huggingface_binary_masks_are_mapped(monkeypatch):
    stream = io.BytesIO()
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(stream, format="PNG")
    mask = base64.b64encode(stream.getvalue()).decode()

    def handler(request):
        assert request.headers["content-type"] == "image/png"
        assert request.content.startswith(b"\x89PNG")
        return httpx.Response(200, json=[{"label": "liver", "mask": mask, "score": 0.9}])

    mock_endpoint(monkeypatch, handler)
    result = RemoteSegmenter("huggingface").predict(
        np.zeros((2, 2, 3), dtype=np.float32),
        LABELS,
        "",
        model("huggingface", label_map={"liver": 1}),
    )
    np.testing.assert_array_equal(result.mask, [[0, 1], [1, 0]])


def test_responses_polygon_adapter_and_explicit_2d_boundary(monkeypatch):
    def handler(request):
        data = json.loads(request.content)
        assert data["model"] == "configured-vision-model"
        assert data["text"]["format"]["strict"] is True
        assert data["input"][0]["content"][1]["type"] == "input_image"
        polygons = {
            "polygons": [
                {"label_id": 1, "points": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 0, "y": 3}]}
            ]
        }
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(polygons)}],
                    }
                ]
            },
        )

    mock_endpoint(monkeypatch, handler)
    provider = RemoteSegmenter("openai-polygons")
    registered = model("openai-polygons", model="configured-vision-model")
    result = provider.predict(np.zeros((4, 4, 3), dtype=np.float32), LABELS, "liver", registered)
    assert result.mask[0, 0] == 1 and result.mask[3, 3] == 0
    with pytest.raises(DomainError, match="2D"):
        provider.predict(np.zeros((4, 4, 4, 1), dtype=np.float32), LABELS, "", registered)


@pytest.mark.parametrize(
    "provider_model,reasoning_effort,max_tokens_field",
    [
        ("switchyard/openai/gpt-5.6-sol", "high", "max_completion_tokens"),
        ("azure/openai/gpt-6-astra", "high", "max_completion_tokens"),
        ("azure/anthropic/claude-opus-5", None, "max_completion_tokens"),
        ("my-custom-gemini-vision-model", None, "max_tokens"),
    ],
)
def test_chat_polygon_adapter_uses_exact_model_and_environment_key(
    monkeypatch, provider_model, reasoning_effort, max_tokens_field
):
    monkeypatch.setenv("NV_INFERENCE_API_KEY", "test-key")

    def handler(request):
        data = json.loads(request.content)
        assert data["model"] == provider_model
        assert request.headers["authorization"] == "Bearer test-key"
        assert data["response_format"]["json_schema"]["strict"] is True
        assert data[max_tokens_field] == 2048
        assert len({"max_tokens", "max_completion_tokens"} & data.keys()) == 1
        if reasoning_effort:
            assert data["reasoning_effort"] == reasoning_effort
        else:
            assert "reasoning_effort" not in data
        content = data["messages"][1]["content"]
        assert json.loads(content[0]["text"])["width"] == 5
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        polygons = {
            "polygons": [
                {
                    "label_id": 1,
                    "points": [
                        {"x": 1, "y": 0},
                        {"x": 4, "y": 0},
                        {"x": 1, "y": 2},
                    ],
                }
            ]
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(polygons),
                            "refusal": None,
                        },
                    }
                ]
            },
        )

    mock_endpoint(monkeypatch, handler)
    result = RemoteSegmenter("openai-chat-polygons").predict(
        np.zeros((3, 5, 1), dtype=np.float32),
        LABELS,
        "liver boundary",
        model(
            "openai-chat-polygons",
            model=provider_model,
            token_env="NV_INFERENCE_API_KEY",
            max_output_tokens=2048,
            reasoning_effort=reasoning_effort,
            max_tokens_field=max_tokens_field,
        ),
    )
    assert result.mask.shape == (3, 5)
    assert result.mask[0, 4] == 1 and result.mask[2, 4] == 0


@pytest.mark.parametrize(
    "choice",
    [
        {"finish_reason": "length", "message": {"content": '{"polygons":[]}'}},
        {"finish_reason": "stop", "message": {"content": None, "refusal": "Unable"}},
        {
            "finish_reason": "stop",
            "message": {
                "content": (
                    '{"polygons":[{"label_id":1,"points":[{"x":true,"y":0},'
                    '{"x":1,"y":0},{"x":1,"y":1}]}]}'
                )
            },
        },
        {
            "finish_reason": "stop",
            "message": {
                "content": (
                    '{"polygons":[{"label_id":1,"points":[{"x":4,"y":0},'
                    '{"x":1,"y":0},{"x":1,"y":1}]}]}'
                )
            },
        },
    ],
)
def test_chat_rejects_incomplete_refused_or_invalid_geometry(monkeypatch, choice):
    mock_endpoint(monkeypatch, lambda request: httpx.Response(200, json={"choices": [choice]}))
    with pytest.raises(DomainError, match="mask contract"):
        RemoteSegmenter("openai-chat-polygons").predict(
            np.zeros((2, 2, 1), dtype=np.float32),
            LABELS,
            "",
            model("openai-chat-polygons", model="configured-model"),
        )


@pytest.mark.parametrize("provider", ["openai-chat-polygons", "openai-polygons"])
def test_captured_polygon_on_crop_boundary_is_accepted_without_extra_requests(
    monkeypatch, provider
):
    # A polygon from the public CMU crop: y=270 is its bottom edge, not an array index.
    polygons = json.loads((Path(__file__).parent / "fixtures/polygon_crop_edge.json").read_text())
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        specification = (
            body["response_format"]["json_schema"]["schema"]
            if provider == "openai-chat-polygons"
            else body["text"]["format"]["schema"]
        )
        point = specification["$defs"]["Point"]["properties"]
        assert point["x"]["minimum"] == 0 and point["x"]["maximum"] == 394
        assert point["y"]["minimum"] == 0 and point["y"]["maximum"] == 270
        message = json.dumps(polygons)
        payload = (
            {"choices": [{"finish_reason": "stop", "message": {"content": message}}]}
            if provider == "openai-chat-polygons"
            else {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": message}]}
                ]
            }
        )
        return httpx.Response(200, json=payload)

    mock_endpoint(monkeypatch, handler)
    prediction = RemoteSegmenter(provider).predict(
        np.zeros((270, 394, 3), dtype=np.float32),
        LABELS,
        "segment the crop",
        model(provider, model="fixture"),
    )
    assert prediction.mask.shape == (270, 394)
    assert prediction.mask[269, 75] == 1
    assert prediction.mask[256, 75] == 0 and prediction.mask[260, 70] == 0
    assert len(calls) == 1


def test_continuous_polygon_edges_rasterize_into_exact_image_extent():
    from monailabel.providers.polygons import mask_from_polygons

    mask = mask_from_polygons(
        {
            "polygons": [
                {
                    "label_id": 1,
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 5, "y": 0},
                        {"x": 5, "y": 3},
                        {"x": 0, "y": 3},
                    ],
                }
            ]
        },
        (3, 5),
        [0, 1],
    )
    np.testing.assert_array_equal(mask, np.ones((3, 5), dtype=np.uint8))


@pytest.mark.parametrize("point", [{"x": 5.01, "y": 1}, {"x": 1, "y": 3.01}])
def test_beyond_crop_edge_is_rejected_with_safe_geometry_diagnostic(monkeypatch, point):
    polygons = {
        "polygons": [{"label_id": 1, "points": [{"x": 0, "y": 0}, point, {"x": 1, "y": 1}]}]
    }
    mock_endpoint(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(polygons)}}]
            },
        ),
    )
    with pytest.raises(DomainError, match="outside the 5 × 3 input image") as error:
        RemoteSegmenter("openai-chat-polygons").predict(
            np.zeros((3, 5, 1), dtype=np.float32),
            LABELS,
            "",
            model("openai-chat-polygons", model="fixture"),
        )
    assert "No partial mask was applied" in str(error.value)


@pytest.mark.parametrize(
    "content", ["sensitive-provider-body", '{"polygons":[{"label_id":"private-value"}]}']
)
def test_polygon_parse_diagnostics_do_not_echo_response_content(monkeypatch, content):
    mock_endpoint(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        ),
    )
    with pytest.raises(DomainError) as error:
        RemoteSegmenter("openai-chat-polygons").predict(
            np.zeros((3, 5, 1), dtype=np.float32),
            LABELS,
            "",
            model("openai-chat-polygons", model="fixture"),
        )
    assert error.value.code == "provider_output_invalid"
    assert "sensitive" not in str(error.value) and "private" not in str(error.value)
