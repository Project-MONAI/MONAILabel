import io
import threading

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from monailabel.core.models import ImageRegion, Proposal
from monailabel.core.ports import Prediction


@pytest.fixture
def region_setup(client, http):
    project = client.post(
        "/api/projects",
        {
            "name": "Selected region",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Nuclei", "color": "#00ff00"},
                {"id": 2, "name": "Tissue", "color": "#ff0000"},
            ],
        },
    )
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    content = io.BytesIO()
    Image.fromarray(pixels).save(content, format="PNG")
    asset = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "fixture.png", "group_id": "source"},
        content=content.getvalue(),
    ).json()
    model = client.post(
        f"/api/projects/{project['id']}/models",
        {
            "name": "Fixture model",
            "provider": "http-mask",
            "label_ids": [0, 1, 2],
            "config": {"url": "http://unused.test/predict"},
        },
    )
    return project, asset, model, pixels


def context_for(asset, model, region):
    return {"asset_id": asset["id"], "model_id": model["id"], "image_region": region}


@pytest.mark.parametrize("irregular", [False, True])
def test_selected_region_only_sends_crop_and_preserves_other_pixels(
    client, http, region_setup, irregular
):
    project, asset, model, pixels = region_setup
    before = np.zeros((12, 16), dtype=np.uint8)
    before[0:2, 0:3] = 1
    before[9, 11] = 2
    before[3:7, 4:9] = 1
    before[3, 4] = 2  # A preserved class within the crop, outside the new prediction.
    client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": 0,
            "mask": before.tolist(),
            "covered_labels": [0, 1, 2],
        },
    )
    region = {"x": 4, "y": 3, "width": 5, "height": 4}
    allowed = np.ones((4, 5), dtype=bool)
    if irregular:
        region["runs"] = [[1, 4], [6, 9], [11, 14]]
        allowed[:] = False
        allowed[:3, 1:4] = True
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            calls.append(image.copy())
            result = np.zeros((4, 5), dtype=np.uint8)
            result[1:3, 1:3] = 1
            result[-1, -1] = 1  # Outside the irregular footprint, must be clipped.
            return Prediction(result)

    http.app.state.services.models.providers["http-mask"] = Segmenter()
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": "annotate the selected region or box for nuclei segmentation",
            "context": context_for(asset, model, region),
        },
    )
    assert "selected 5 × 4 pixel region" in reply["message"]
    outcome = client.wait(reply["job_id"])
    proposal = client.get("/api/proposals/" + outcome["proposal_id"])
    assert proposal["image_region"]["x"] == 4
    result = np.frombuffer(
        http.get("/api/proposals/" + proposal["id"] + "/mask.bin").content, dtype=np.uint8
    ).reshape(12, 16)
    expected_input = pixels[3:7, 4:9].astype(np.float32) / 255
    if irregular:
        expected_input = np.where(allowed[..., None], expected_input, 0)
    np.testing.assert_array_equal(calls[0], expected_input)
    outside = np.ones((12, 16), dtype=bool)
    outside[3:7, 4:9] = ~allowed
    np.testing.assert_array_equal(result[outside], before[outside])
    assert result[4, 5] == 1 and result[5, 6] == 1 and result[3, 4] == 2
    assert result[6, 8] == 1  # Existing foreground outside the irregular region is retained.
    assert client.get("/api/assets/" + asset["id"])["revision"] == 1


@pytest.mark.parametrize("region", [None, {"x": 15, "y": 3, "width": 3, "height": 2}])
def test_missing_or_outside_selection_never_falls_back_to_whole_image(
    http, client, region_setup, region
):
    project, asset, model, _ = region_setup
    response = http.post(
        f"/api/projects/{project['id']}/assistant",
        json={
            "message": "annotate nuclei inside this region",
            "context": context_for(asset, model, region),
        },
    )
    assert response.status_code == 422
    assert client.get(f"/api/projects/{project['id']}/jobs") == []


@pytest.mark.parametrize(
    "runs", [[], [[0, 21]], [[-1, 2]], [[3, 2]], [[0, 5], [4, 8]], [[True, 2]]]
)
def test_invalid_footprints_are_rejected(runs):
    with pytest.raises(ValidationError):
        ImageRegion(x=0, y=0, width=5, height=4, runs=runs)


def test_region_cancellation_publishes_no_proposal(client, http, region_setup):
    project, asset, model, _ = region_setup
    entered, release = threading.Event(), threading.Event()

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            entered.set()
            assert release.wait(10)
            return Prediction(np.ones(image.shape[:2], dtype=np.uint8))

    http.app.state.services.models.providers["http-mask"] = Segmenter()
    job = client.post(
        "/api/assets/" + asset["id"] + "/annotate",
        {
            "model_id": model["id"],
            "image_region": {"x": 3, "y": 4, "width": 5, "height": 4},
        },
    )
    try:
        assert entered.wait(5)
        client.post("/api/jobs/" + job["id"] + "/cancel")
    finally:
        release.set()
    with pytest.raises(RuntimeError):
        client.wait(job["id"])
    assert http.app.state.services.store.list(Proposal, project["id"]) == []


def test_provided_region_is_one_crop_even_with_default_tiling_context(client, http, region_setup):
    project, asset, model, _ = region_setup
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            calls.append(image.shape)
            return Prediction(np.zeros(image.shape[:2], dtype=np.uint8))

    http.app.state.services.models.providers["http-mask"] = Segmenter()
    context = context_for(asset, model, {"x": 4, "y": 3, "width": 5, "height": 4})
    context["image_tiling"] = {"tile_size": 64, "overlap": 8}
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": "run nuclei segmentation and use tile size 512",
            "context": context,
        },
    )
    result = client.wait(reply["job_id"])
    proposal = client.get("/api/proposals/" + result["proposal_id"])
    assert calls == [(4, 5, 3)] and proposal["image_tiling"] is None
    assert "one crop, without tiling" in reply["message"]


@pytest.mark.parametrize(
    "message",
    [
        "run nuclie on whole image and use tile size 512",
        "run nuclei on the whole slide using tile size 512",
        "annotate the full image for nuclei using Fixture model and use tile size 512",
        "annotate nuclei on the full image with tile size 512x512 using Fixture model",
    ],
)
def test_whole_image_prompt_tile_size_overrides_viewer_default(client, http, region_setup, message):
    project, asset, model, _ = region_setup
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            calls.append(image.shape)
            return Prediction(np.zeros(image.shape[:2], dtype=np.uint8))

    http.app.state.services.models.providers["http-mask"] = Segmenter()
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": message,
            "context": {
                "asset_id": asset["id"],
                "model_id": model["id"],
                "image_tiling": {"tile_size": 256, "overlap": 32},
            },
        },
    )
    result = client.wait(reply["job_id"])
    proposal = client.get("/api/proposals/" + result["proposal_id"])
    assert proposal["image_region"] is None
    assert proposal["image_tiling"] == {"tile_size": 512, "overlap": 32}
    assert proposal["label_ids"] == [1]  # The spelling variant must not select Tissue too.
    assert "512 × 512" in reply["message"] and len(calls) == 1


@pytest.mark.parametrize("size", ["abc", "0", "32", "4096", "512.5", "512x256", "-512"])
def test_invalid_prompt_tile_size_creates_no_job(client, http, region_setup, size):
    project, asset, model, _ = region_setup
    response = http.post(
        f"/api/projects/{project['id']}/assistant",
        json={
            "message": f"run nuclei on the whole slide using tile size {size}",
            "context": {"asset_id": asset["id"], "model_id": model["id"]},
        },
    )
    assert response.status_code == 422
    assert "64 to 2048" in response.json()["detail"]
    assert client.get(f"/api/projects/{project['id']}/jobs") == []


def test_api_cannot_combine_region_and_tiling():
    from monailabel.core.models import AnnotateRequest

    with pytest.raises(ValidationError, match="one crop"):
        AnnotateRequest(
            image_region=ImageRegion(x=0, y=0, width=100, height=100),
            image_tiling={"tile_size": 64, "overlap": 8},
        )


def test_full_image_tiles_preserve_resolution_edges_and_stitch_once(client, http, region_setup):
    from monailabel.core.models import ImageTiling
    from monailabel.core.tiling import image_tiles

    project, asset, model, _ = region_setup
    pixels = np.zeros((105, 137, 3), dtype=np.uint8)
    pixels[25:76, 30:110, 0] = 200  # Target crosses several tile boundaries.
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="PNG")
    asset = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "tiled.png", "group_id": "tiled"},
        content=out.getvalue(),
    ).json()
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            calls.append(image.shape)
            assert image.shape[0] <= 64 and image.shape[1] <= 64
            return Prediction((image[..., 0] > 0.5).astype(np.uint8))

    http.app.state.services.models.providers["http-mask"] = Segmenter()
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": "annotate the full image for nuclei",
            "context": {
                "asset_id": asset["id"],
                "model_id": model["id"],
                "image_tiling": {"tile_size": 64, "overlap": 8},
            },
        },
    )
    result = client.wait(reply["job_id"])
    assert "9 native-resolution tiles" in reply["message"] and len(calls) == 9
    mask = np.frombuffer(
        http.get("/api/proposals/" + result["proposal_id"] + "/mask.bin").content, dtype=np.uint8
    ).reshape(105, 137)
    np.testing.assert_array_equal(mask, pixels[..., 0] > 0)
    coverage = np.zeros((105, 137), dtype=np.uint8)
    for tile in image_tiles(105, 137, ImageTiling(tile_size=64, overlap=8)):
        w = tile.write
        coverage[w.y : w.y + w.height, w.x : w.x + w.width] += 1
    assert np.all(coverage == 1)


def test_tiled_failure_does_not_publish_partial_labels(client, http, region_setup):
    project, asset, model, _ = region_setup
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            calls.append(1)
            from monailabel.core.errors import DomainError

            raise DomainError("Fixture provider failed")

    http.app.state.services.models.providers["http-mask"] = Segmenter()
    job = client.post(
        "/api/assets/" + asset["id"] + "/annotate",
        {
            "model_id": model["id"],
            "image_tiling": {"tile_size": 64, "overlap": 8},
        },
    )
    with pytest.raises(RuntimeError, match="Tile 1"):
        client.wait(job["id"])
    assert http.app.state.services.store.list(Proposal, project["id"]) == []
