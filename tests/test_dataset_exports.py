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

import gzip
import hashlib
import io
import json
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest
import test_image_regions
import test_videos
from PIL import Image

region_setup = test_image_regions.region_setup
clip = test_videos.clip
video = test_videos.video


@pytest.mark.parametrize("mode, revisions", [("latest", [2]), ("all", [1, 2])])
def test_dataset_export_preserves_geometry_versions_and_manifest(http, client, mode, revisions):
    project = client.post(
        "/api/projects",
        {
            "name": "Export test",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Spleen", "color": "#ff0000"},
            ],
        },
    )
    prefix = f"/api/projects/{project['id']}"
    affine = np.array([[0, -1.2, 0, 10], [0.8, 0, 0, -30], [0, 0, 2.5, 7], [0, 0, 0, 1]])
    image = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    content = gzip.compress(nib.Nifti1Image(image, affine).to_bytes())
    asset = http.post(
        prefix + "/assets/upload",
        params={"name": "original.nii.gz", "group_id": "patient-42"},
        content=content,
    ).json()
    masks = []
    for revision in range(2):
        mask = np.zeros((3, 4, 5), np.uint8)
        mask[revision, 1:3, :] = 1
        masks.append(mask)
        response = http.post(
            f"/api/assets/{asset['id']}/review-mask",
            params={"base_revision": revision, "covered_labels": [1]},
            content=mask.tobytes(),
        )
        assert response.status_code == 201, response.text
    annotation = response.json()
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    request = {"asset_ids": [asset["id"], asset["id"]], "annotations": mode}
    response = http.post(
        prefix + "/exports", json=request, headers={"Idempotency-Key": "export-once"}
    )
    assert response.status_code == 202, response.text
    job = response.json()
    result = client.wait(job["id"])
    assert result["asset_count"] == 1 and result["annotation_count"] == len(revisions)
    assert (
        http.post(
            prefix + "/exports", json=request, headers={"Idempotency-Key": "export-once"}
        ).json()["id"]
        == job["id"]
    )
    download = http.get(f"/api/jobs/{job['id']}/export")
    assert download.status_code == 200 and download.headers["content-type"] == "application/zip"
    with ZipFile(io.BytesIO(download.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["classes"] == project["labels"]
        sample = manifest["assets"][0]
        assert sample["group_id"] == "patient-42" and sample["revision"] == 2
        assert archive.read(sample["image"]) == content
        assert np.allclose(sample["affine"], affine)
        assert [a["revision"] for a in sample["annotations"]] == revisions
        assert sample["annotations"][-1]["review"]["verdict"] == "accepted"
        for annotation in sample["annotations"]:
            mask = nib.Nifti1Image.from_bytes(archive.read(annotation["file"]))
            assert np.allclose(mask.affine, affine)
            assert np.array_equal(np.asarray(mask.dataobj), masks[annotation["revision"] - 1])
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            assert len(data) == entry["size_bytes"]
        assert all(
            not name.startswith("/") and ".." not in name.split("/") for name in archive.namelist()
        )
    other = client.post("/api/projects", {"name": "Other"})
    assert http.post(f"/api/projects/{other['id']}/exports", json=request).status_code == 403
    assert client.get(f"/api/projects/{project['id']}/assets")[0]["revision"] == 2


def exported(http, client, project_id, asset_id, mode="latest"):
    job = client.post(
        f"/api/projects/{project_id}/exports",
        {"asset_ids": [asset_id], "annotations": mode},
    )
    client.wait(job["id"])
    response = http.get(f"/api/jobs/{job['id']}/export")
    assert response.status_code == 200
    archive = ZipFile(io.BytesIO(response.content))
    return archive, json.loads(archive.read("manifest.json"))


def test_export_region_retains_review_coverage(http, client, region_setup):
    project, asset, _, _ = region_setup
    region = {"x": 1, "y": 2, "width": 4, "height": 3}
    draft = np.ones((12, 16), dtype=np.uint8)
    client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "regions": [region], "covered_labels": [0, 1], "mask": draft.tolist()},
    )
    unit = client.get(f"/api/projects/{project['id']}/review-units")[0]
    client.post(
        f"/api/review-units/{unit['id']}/decision",
        {"base_revision": unit["revision"], "verdict": "accepted"},
    )
    archive, manifest = exported(http, client, project["id"], asset["id"])
    with archive:
        sample = manifest["assets"][0]
        reference = sample["review_units"][0]
        assert reference["scope"] == {"kind": "region", "region": region | {"runs": None}}
        assert reference["review"]["verdict"] == "accepted"
        mask = np.asarray(Image.open(io.BytesIO(archive.read(reference["file"]))))
        assert np.count_nonzero(mask) == 12
        submitted = np.asarray(
            Image.open(io.BytesIO(archive.read(sample["annotations"][0]["file"])))
        )
        assert submitted.shape == (12, 16)
        assert np.count_nonzero(submitted) == 12
        assert np.all(submitted[2:5, 1:5] == 1)


def test_export_video_retains_source_frames_tracks_and_versions(http, client, video, clip):
    documents = []
    for revision in range(2):
        document = {
            "tracks": [
                {
                    "id": "instrument",
                    "label_id": 1,
                    "keyframes": [
                        {"frame": 1, "points": [1, 1, 20 + revision, 1, 20, 20, 1, 20]},
                        {"frame": 3, "points": [1, 1, 20, 1, 20, 20, 1, 20], "outside": True},
                    ],
                }
            ]
        }
        documents.append(document)
        client.post(
            f"/api/videos/{video['id']}/review",
            {"base_revision": revision, "document": document},
        )
    for mode, revisions in (("latest", [2]), ("all", [1, 2])):
        archive, manifest = exported(http, client, video["project_id"], video["id"], mode)
        with archive:
            sample = manifest["assets"][0]
            assert archive.read(sample["image"]) == clip
            assert sample["group_id"] == "procedure-1"
            assert sample["video_metadata"]
            assert [a["revision"] for a in sample["annotations"]] == revisions
            assert [u["revision"] for u in sample["review_units"]] == revisions
            for annotation in sample["annotations"]:
                tracks = json.loads(archive.read(annotation["file"]))["tracks"]
                expected = documents[annotation["revision"] - 1]["tracks"][0]
                assert tracks[0]["keyframes"][0]["points"] == expected["keyframes"][0]["points"]
                assert tracks[0]["keyframes"][-1]["outside"]
            for reference in sample["review_units"]:
                assert reference["scope"] == {"kind": "frames", "start": 1, "stop": 3}
                assert json.loads(archive.read(reference["file"]))["tracks"]
