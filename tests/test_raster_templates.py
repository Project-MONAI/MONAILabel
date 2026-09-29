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

"""2D templates preserve masks, review gates and conservative source grouping."""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image


def png(values):
    stream = io.BytesIO()
    Image.fromarray(values).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def raster_fixture(client, http, tmp_path, monkeypatch):
    project = client.post("/api/projects", {"name": "Labeled raster samples"})
    path = tmp_path / "images.zip"
    masks = tmp_path / "masks.zip"
    expected = np.zeros((9, 11), dtype=np.uint8)
    expected[2:6, 3:8] = 255
    with zipfile.ZipFile(path, "w") as archive:
        for patient in range(1, 4):
            for patch in range(1, 3):
                stem = f"{patient:02}_{patch}.png"
                image = np.full((9, 11, 3), patient * 10 + patch, dtype=np.uint8)
                archive.writestr(f"TNBC/Slide_{patient:02}/{stem}", png(image))
                archive.writestr(f"TNBC/GT_{patient:02}/{stem}", png(expected))
        for index in range(2):
            archive.writestr(f"images/frame{index}.jpg", png(np.full((9, 11, 3), index, np.uint8)))
    with zipfile.ZipFile(masks, "w") as archive:
        for index in range(2):
            archive.writestr(
                f"masks/frame{index}.png", png(np.repeat(expected[..., None], 3, axis=2))
            )
    fetched = []

    def fetch(source, context):
        fetched.append(source.id)
        return masks if source.id.endswith("-masks") else path

    monkeypatch.setattr(http.app.state.services.dataset_templates.downloads, "fetch", fetch)
    return "/api/projects/" + project["id"], masks, expected, fetched


def imported(client, prefix, template="tnbc-nuclei", **choices):
    job = client.post(prefix + "/dataset-imports", {"template_id": template, **choices})
    return client.wait(job["id"])


def test_tnbc_patient_split_masks_and_attribution(client, http, raster_fixture):
    prefix, _, expected, _ = raster_fixture
    result = imported(client, prefix, limit=None, include_masks=True, evaluation_percentage=34)
    assert not result["failed"]
    assets = client.get(prefix + "/assets")
    assert len(assets) == 6
    uses = {}
    for asset in assets:
        uses.setdefault(asset["group_id"], set()).add(asset["split"])
        annotation = client.get(f"/api/assets/{asset['id']}/annotations")[0]
        mask = np.frombuffer(
            http.get(f"/api/annotations/{annotation['id']}/mask.bin").content, np.uint8
        )
        assert np.array_equal(mask.reshape(expected.shape) > 0, expected > 0)
        assert asset["spatial_shape"] == list(expected.shape)
    assert uses == {"tnbc:01": {"validation"}, "tnbc:02": {"validation"}, "tnbc:03": {"pool"}}
    decisions = client.get(prefix + "/decisions")
    assert len(decisions) == 4 and all(d["verdict"] == "accepted" for d in decisions)
    assert {d["asset_id"] for d in decisions} == set(result["evaluation_asset_ids"])
    job = client.get(prefix + "/jobs")[-1]
    assert job["request"]["license"] == "CC BY 4.0"
    assert "Naylor" in job["request"]["citation"]
    # Repeating the same import keeps prior annotations and does not multiply assets.
    again = imported(client, prefix, limit=None, include_masks=True, evaluation_percentage=34)
    assert not again["failed"]
    assert set(again["asset_ids"]) == set(result["asset_ids"])
    assert client.get(prefix + "/decisions") == decisions


def test_tnbc_quick_start_visits_different_patients(client, raster_fixture):
    prefix, _, _, _ = raster_fixture
    result = imported(client, prefix, limit=3)
    assert not result["failed"]
    assert {a["group_id"] for a in client.get(prefix + "/assets")} == {
        "tnbc:01",
        "tnbc:02",
        "tnbc:03",
    }
    assert not result["annotation_ids"]


def test_instruments_mask_archive_and_evaluation_reservation(client, http, raster_fixture):
    prefix, _, _, fetched = raster_fixture
    result = imported(
        client, prefix, "kvasir-instrument", limit=1, split="validation", include_masks=True
    )
    assert not result["failed"]
    assert fetched == ["kvasir-instrument", "kvasir-instrument-masks"]
    asset = client.get(f"/api/assets/{result['asset_ids'][0]}")
    assert asset["group_id"] == "kvasir-instrument:unknown-procedures"
    assert asset["split"] == "validation"
    # A different frame is still in the reserved source group.
    result = imported(client, prefix, "kvasir-instrument", offset=1, limit=1, split="train")
    assert not result["asset_ids"] and result["failed"]
    assert fetched[-1] == "kvasir-instrument"  # Images-only import does not fetch masks.
    response = http.post(
        prefix + "/dataset-imports",
        json={
            "template_id": "kvasir-instrument",
            "evaluation_percentage": 20,
        },
    )
    assert response.status_code == 422
    assert "patient/procedure" in response.text


@pytest.mark.parametrize("broken", ["shape", "color", "values", "missing"])
def test_bad_raster_mask_publishes_no_asset(client, raster_fixture, broken):
    prefix, path, _, _ = raster_fixture
    with zipfile.ZipFile(path, "w") as archive:
        if broken != "missing":
            shape = (8, 11) if broken == "shape" else (9, 11)
            values = np.zeros(shape, np.uint8)
            if broken == "values":
                values[0, 0] = 2
            if broken == "color":
                values = np.repeat(values[..., None], 3, axis=2)
                values[0, 0, 1] = 255
            archive.writestr("masks/frame0.png", png(values))
    result = imported(client, prefix, "kvasir-instrument", limit=1, include_masks=True)
    assert result["asset_ids"] == [] and len(result["failed"]) == 1
    assert client.get(prefix + "/assets") == []
