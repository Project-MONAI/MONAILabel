"""Public template imports keep label identity, source separation and review gates."""

import gzip
import io
import json
import tarfile
import zipfile

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.errors import DomainError
from monailabel.server.dataset_downloads import Archive
from monailabel.server.dataset_templates import select_channel


def nifti(values, affine=None):
    return gzip.compress(
        nib.Nifti1Image(values, np.eye(4) if affine is None else affine).to_bytes()
    )


@pytest.fixture
def template_fixture(client, http, tmp_path, monkeypatch):
    project = client.post(
        "/api/projects",
        {
            "name": "Imported references",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Liver"},
                {"id": 4, "name": "Spleen"},
            ],
        },
    )
    shape = (8, 9, 10)
    image = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2:4, 3:6, 2:7] = 1
    path = tmp_path / "dataset.tar"

    def write(entries):
        with tarfile.open(path, "w") as archive:
            for name, content in entries.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))

    entries = {
        "./Task09_Spleen/dataset.json": json.dumps(
            {"labels": {"0": "background", "1": "spleen"}}
        ).encode(),
        "Task09_Spleen/imagesTr/spleen_1.nii.gz": nifti(image),
        "Task09_Spleen/labelsTr/spleen_1.nii.gz": nifti(mask),
        "Task09_Spleen/imagesTr/spleen_2.nii.gz": nifti(image + 1),
        "Task09_Spleen/labelsTr/spleen_2.nii.gz": nifti(mask),
        "Task09_Spleen/imagesTr/._spleen_2.nii.gz": b"mac metadata",
    }
    write(entries)
    monkeypatch.setattr(
        http.app.state.services.dataset_templates.downloads, "fetch", lambda *args: path
    )
    return "/api/projects/" + project["id"], path, entries, write, mask


def imported(client, prefix, **choices):
    job = client.post(
        prefix + "/dataset-imports", {"template_id": "Task09_Spleen", "limit": 1, **choices}
    )
    return client.wait(job["id"])


def test_catalog_and_images_only(client, http, template_fixture):
    prefix, _, _, _, _ = template_fixture
    catalog = client.get(prefix + "/dataset-templates")
    assert len([x for x in catalog if x["id"].startswith("Task")]) == 10
    assert {x["category"] for x in catalog} == {"Radiology", "Pathology", "Video"}
    assert "url" not in catalog[0] and "checksum" not in catalog[0]
    before = client.get(prefix)["labels"]
    result = imported(client, prefix)
    assert result["failed"] == []
    assert result["annotation_ids"] == []
    assets = client.get(prefix + "/assets")
    assert len(assets) == 1 and assets[0]["annotation_id"] is None
    assert client.get(prefix)["labels"] == before
    again = imported(client, prefix)
    assert again["asset_ids"] == result["asset_ids"]
    assert len(client.get(prefix + "/assets")) == 1


def test_all_samples_and_explicit_subset(client, template_fixture):
    prefix, _, entries, write, _ = template_fixture
    # More than the five-sample default makes an accidentally retained limit observable.
    for index in range(3, 8):
        entries[f"Task09_Spleen/imagesTr/spleen_{index}.nii.gz"] = nifti(
            np.full((8, 9, 10), index, dtype=np.int16)
        )
    write(entries)
    subset = imported(client, prefix, offset=2, limit=2)
    assert len(subset["asset_ids"]) == 2
    result = imported(client, prefix, limit=None)
    assert result["failed"] == []
    assert len(result["asset_ids"]) == 7
    assert set(subset["asset_ids"]) <= set(result["asset_ids"])
    assert len(client.get(prefix + "/assets")) == 7


def test_masks_remapped_pending_review_and_retry_preserves_edits(client, http, template_fixture):
    prefix, _, _, _, mask = template_fixture
    result = imported(client, prefix, include_masks=True, split="train")
    assert result["failed"] == []
    annotation = client.get("/api/assets/" + result["asset_ids"][0] + "/annotations")[0]
    assert annotation["covered_labels"] == [0, 4]
    actual = np.frombuffer(
        http.get("/api/annotations/" + annotation["id"] + "/mask.bin").content, dtype=np.uint8
    ).reshape(mask.shape)
    np.testing.assert_array_equal(actual, mask * 4)
    assert client.get(prefix + "/decisions") == []
    before = client.get(prefix + "/assets")
    again = imported(client, prefix, include_masks=True, split="train")
    assert again["annotation_ids"] == []
    assert client.get(prefix + "/assets") == before
    validation = imported(client, prefix, include_masks=True, split="validation")
    assert validation["asset_ids"] == []
    assert "assigned to training" in validation["failed"][0]["error"]
    second = imported(client, prefix, include_masks=True, split="validation", offset=1)
    assert len(second["asset_ids"]) == 1
    assert client.get("/api/assets/" + second["asset_ids"][0])["group_id"] != before[0]["group_id"]


@pytest.mark.parametrize("include_masks", [False, True])
def test_recreated_project_import_does_not_restore_deleted_annotations(
    client, http, template_fixture, include_masks
):
    prefix, path, _, _, source_mask = template_fixture
    original_project = client.get(prefix)
    imported_asset = imported(client, prefix)["asset_ids"][0]
    edited = client.post(
        f"/api/assets/{imported_asset}/review",
        {
            "base_revision": 0,
            "mask": np.full(source_mask.shape, 4, dtype=np.uint8).tolist(),
            "covered_labels": [0, 4],
        },
    )
    response = http.request("DELETE", prefix, json={"confirmation_name": original_project["name"]})
    assert response.status_code == 200
    assert path.exists()  # The source archive remains reusable after project deletion.
    assert http.get(f"/api/annotations/{edited['id']}/mask.bin").status_code == 404

    recreated = client.post(
        "/api/projects",
        {"name": original_project["name"], "labels": original_project["labels"]},
    )
    assert recreated["id"] != original_project["id"]
    result = imported(
        client,
        f"/api/projects/{recreated['id']}",
        include_masks=include_masks,
        split="validation" if include_masks else "pool",
    )
    assert result["failed"] == []
    asset = client.get(f"/api/assets/{result['asset_ids'][0]}")
    assert asset["id"] != imported_asset
    annotations = client.get(f"/api/assets/{asset['id']}/annotations")
    if include_masks:
        assert len(annotations) == 1
        annotation = annotations[0]
        assert annotation["id"] != edited["id"]
        assert annotation["revision"] == 1
        actual = np.frombuffer(
            http.get(f"/api/annotations/{annotation['id']}/mask.bin").content, dtype=np.uint8
        ).reshape(source_mask.shape)
        np.testing.assert_array_equal(actual, source_mask * 4)
    else:
        assert asset["revision"] == 0
        assert asset["annotation_id"] is None
        assert annotations == []


@pytest.mark.parametrize("broken", ["affine", "missing", "fractional", "unknown_id"])
def test_bad_reference_never_publishes_annotation(client, http, template_fixture, broken):
    prefix, _, entries, write, mask = template_fixture
    name = "Task09_Spleen/labelsTr/spleen_1.nii.gz"
    if broken == "affine":
        entries[name] = nifti(mask, np.diag([2, 2, 2, 1]))
    elif broken == "missing":
        del entries[name]
    elif broken == "fractional":
        entries[name] = nifti(mask.astype(np.float32) * 1.5)
    else:
        entries[name] = nifti(mask * 9)
    write(entries)
    result = imported(client, prefix, include_masks=True)
    assert result["asset_ids"] == []
    assert len(result["failed"]) == 1
    assert client.get(prefix + "/assets") == []


def test_totalsegmentator_subset_and_anatomical_colors(client, http, template_fixture):
    prefix, path, _, _, mask = template_fixture
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/s001/ct.nii.gz", nifti(mask.astype(np.int16)))
        archive.writestr("data/s001/segmentations/kidney_left.nii.gz", nifti(mask))
    result = imported(
        client,
        prefix,
        template_id="totalsegmentator-small",
        include_masks=True,
        targets=["kidney_left"],
    )
    assert result["failed"] == []
    project = client.get(prefix)
    label = project["labels"][-1]
    assert label["name"] == "Kidney left"
    from monailabel.core.colors import default_color

    assert label["color"] == default_color("left kidney", label["id"])
    asset = client.get("/api/assets/" + result["asset_ids"][0])
    assert asset["name"] == "s001.nii.gz"
    assert asset["group_id"] == "totalsegmentator:s001"


def test_invalid_choices_and_permissions_fail_before_download(
    client, http, template_fixture, monkeypatch
):
    prefix, _, _, _, _ = template_fixture

    def forbidden(*args):
        raise AssertionError("Must not download")

    monkeypatch.setattr(http.app.state.services.dataset_templates.downloads, "fetch", forbidden)
    for choice in [
        {"template_id": "no-such-source"},
        {"template_id": "openslide-cmu-small", "include_masks": True},
        {"template_id": "Task09_Spleen", "section": "test", "include_masks": True},
        {"template_id": "totalsegmentator-small", "include_masks": True},
        {"template_id": "totalsegmentator-small", "include_masks": True, "targets": ["unknown"]},
    ]:
        assert http.post(prefix + "/dataset-imports", json=choice).status_code == 422
    user = client.post("/api/auth/users", {"username": "viewer", "password": "test-password-1234"})
    client.request("PUT", prefix + "/members", {"user_id": user["id"], "roles": ["annotator"]})
    http.post("/api/auth/login", json={"username": "viewer", "password": "test-password-1234"})
    assert (
        http.post(prefix + "/dataset-imports", json={"template_id": "Task09_Spleen"}).status_code
        == 403
    )


def test_archive_rejects_traversal_and_bounds_reads(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside", b"secret")
    with pytest.raises(DomainError, match="unsafe"):
        Archive(path)
    assert not (tmp_path.parent / "outside").exists()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("large", b"12345")
    archive = Archive(path)
    try:
        with pytest.raises(DomainError, match="limit"):
            archive.read("large", 4)
    finally:
        archive.close()


def test_modality_selection_preserves_geometry():
    values = np.zeros((3, 4, 5, 2), dtype=np.int16)
    values[..., 1] = 17
    affine = np.diag([2, 3, 4, 1])
    output = nib.Nifti1Image.from_bytes(gzip.decompress(select_channel(nifti(values, affine), 1)))
    np.testing.assert_array_equal(output.get_fdata(), values[..., 1])
    np.testing.assert_array_equal(output.affine, affine)
    with pytest.raises(DomainError, match="absent"):
        select_channel(nifti(values, affine), 2)


def chat_tool(name, **arguments):
    from uuid import uuid4

    from monailabel.core.chat import ChatMessage, ToolCall

    return ChatMessage(
        role="assistant", tool_calls=[ToolCall(id=uuid4().hex, name=name, arguments=arguments)]
    )


def test_chat_catalog_then_ten_images_is_a_job_and_request_retry_is_idempotent(
    client, http, template_fixture
):
    from uuid import uuid4

    prefix, _, entries, write, _ = template_fixture
    for index in range(3, 13):
        entries[f"Task09_Spleen/imagesTr/spleen_{index}.nii.gz"] = nifti(
            np.full((8, 9, 10), index, dtype=np.int16)
        )
    write(entries)
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        chat_tool("inspect_workspace", collection="dataset_templates"),
        chat_tool("import_dataset_template", template_id="Task09_Spleen", limit=10),
    ]
    body = {
        "message": "Import 10 images from Sample Medical Decathlon Dataset for spleen "
        "to start annotation",
        "request_id": uuid4().hex,
    }
    reply = client.post(prefix + "/assistant", body)
    assert reply["tools"] == ["inspect_workspace", "import_dataset_template"]
    assert reply["data"] == {"page": "datasets", "template_id": "Task09_Spleen"}
    assert reply["job_id"] and "Started importing up to 10 images only" in reply["message"]
    assert "Task09_Spleen" in planner.calls[1][0][-1].content
    assert '"url":' not in planner.calls[1][0][-1].content
    job = client.get("/api/jobs/" + reply["job_id"])
    assert job["request"]["limit"] == 10
    assert job["request"]["split"] == "pool" and job["request"]["section"] == "training"
    assert job["request"]["include_masks"] is False
    result = client.wait(reply["job_id"])
    assert len(result["asset_ids"]) == 10 and result["failed"] == []
    assert result["annotation_ids"] == []
    assert all(a["split"] == "pool" for a in client.get(prefix + "/assets"))
    body["conversation_id"] = reply["conversation_id"]
    assert client.post(prefix + "/assistant", body) == reply
    assert len(client.get(prefix + "/jobs")) == 1
    assert len(planner.calls) == 2


@pytest.mark.parametrize("include_masks,split", [(False, "pool"), (True, "train")])
def test_chat_preserves_requested_masks_and_split(
    client, http, template_fixture, include_masks, split
):
    prefix, _, _, _, _ = template_fixture
    http.app.state.services.assistants.provider.queue = [
        chat_tool(
            "import_dataset_template",
            template_id="Task09_Spleen",
            include_masks=include_masks,
            split=split,
            limit=1,
        )
    ]
    reply = client.post(prefix + "/assistant", {"message": "Import the selected sample"})
    result = client.wait(reply["job_id"])
    assert len(result["annotation_ids"]) == int(include_masks)
    assert client.get(prefix + "/assets")[0]["split"] == split
    assert client.get(prefix + "/decisions") == []


@pytest.mark.parametrize("template", ["unknown", "monuseg"])
def test_chat_invalid_or_external_only_source_starts_no_job(
    client, http, template_fixture, template
):
    prefix, _, _, _, _ = template_fixture
    http.app.state.services.assistants.provider.queue = [
        chat_tool("import_dataset_template", template_id=template, limit=10)
    ]
    response = http.post(prefix + "/assistant", json={"message": "Import those samples"})
    assert response.status_code == 422
    assert client.get(prefix + "/jobs") == []


def test_chat_template_import_requires_manager(client, http, template_fixture):
    prefix, _, _, _, _ = template_fixture
    user = client.post(
        "/api/auth/users", {"username": "annotator", "password": "test-password-1234"}
    )
    client.request("PUT", prefix + "/members", {"user_id": user["id"], "roles": ["annotator"]})
    client.post("/api/auth/login", {"username": "annotator", "password": "test-password-1234"})
    http.app.state.services.assistants.provider.queue = [
        chat_tool("import_dataset_template", template_id="Task09_Spleen", limit=10)
    ]
    response = http.post(prefix + "/assistant", json={"message": "Import those samples"})
    assert response.status_code == 403
    assert client.get(prefix + "/jobs") == []


@pytest.mark.parametrize("form", ["dataset-template", "dicom", "dataset"])
def test_chat_can_open_the_requested_import_chooser_without_importing(
    client, http, template_fixture, form
):
    prefix, _, _, _, _ = template_fixture
    http.app.state.services.assistants.provider.queue = [chat_tool("open_form", form=form)]
    reply = client.post(prefix + "/assistant", {"message": "Show the import choices"})
    assert reply["data"] == {"form": form} and reply["job_id"] is None
    assert client.get(prefix + "/jobs") == []


def test_chat_all_images_uses_unlimited_import_and_keeps_source_selection(
    client, http, template_fixture
):
    prefix, _, entries, write, _ = template_fixture
    for index in range(3, 8):
        entries[f"Task09_Spleen/imagesTs/test_{index}.nii.gz"] = nifti(
            np.full((8, 9, 10), index, dtype=np.int16)
        )
    write(entries)
    http.app.state.services.assistants.provider.queue = [
        chat_tool(
            "import_dataset_template", template_id="Task09_Spleen", all_samples=True, section="test"
        )
    ]
    reply = client.post(prefix + "/assistant", {"message": "Import all spleen test images"})
    job = client.get("/api/jobs/" + reply["job_id"])
    assert job["request"]["limit"] is None and job["request"]["section"] == "test"
    assert "all_samples" not in job["request"]
    result = client.wait(reply["job_id"])
    assert len(result["asset_ids"]) == 5 and result["annotation_ids"] == []
    assert all(a["name"].startswith("test_") for a in client.get(prefix + "/assets"))


def test_chat_evaluation_import_includes_five_labels_reserves_and_reuses_set(
    client, http, template_fixture
):
    prefix, _, entries, write, mask = template_fixture
    for index in range(3, 7):
        entries[f"Task09_Spleen/imagesTr/spleen_{index}.nii.gz"] = nifti(
            np.full(mask.shape, index, dtype=np.int16)
        )
        entries[f"Task09_Spleen/labelsTr/spleen_{index}.nii.gz"] = nifti(mask)
    write(entries)
    http.app.state.services.assistants.provider.queue = [
        chat_tool(
            "import_dataset_template", template_id="Task09_Spleen", split="validation", limit=5
        )
    ]
    reply = client.post(
        prefix + "/assistant",
        {"message": "import 5 samples for evaluation from decathlon spleen dataset"},
    )
    job = client.get("/api/jobs/" + reply["job_id"])
    assert job["request"]["include_masks"] is True
    assert job["request"]["section"] == "training"
    assert "reserved for evaluation" in reply["message"]
    result = client.wait(job["id"])
    assert len(result["asset_ids"]) == len(result["annotation_ids"]) == 5
    assert not result["failed"]
    assets = client.get(prefix + "/assets")
    assert all(a["split"] == "validation" and a["annotation_id"] for a in assets)
    record = client.get(prefix + "/evaluation-sets")[0]
    assert len(record["member_groups"]) == 5 and not record["auto_update"]
    assert result["evaluation_set_id"] == record["id"]
    assert client.get(prefix + "/decisions") == []  # Never manufacture acceptance.
    again = imported(client, prefix, split="validation", limit=5, include_masks=False)
    assert again["asset_ids"] == result["asset_ids"]
    assert client.get(prefix + "/evaluation-sets") == [record]
    assert client.get(prefix + "/assets") == assets
    added = imported(client, prefix, split="validation", offset=5, limit=1)
    assert added["evaluation_set_id"] == record["id"]
    assert len(client.get(prefix + "/evaluation-sets")[0]["member_groups"]) == 6
    rejected = imported(client, prefix, split="train", limit=5)
    assert not rejected["asset_ids"] and len(rejected["failed"]) == 5


def test_evaluation_template_rejects_unlabeled_sources_before_download(
    client, http, template_fixture, monkeypatch
):
    prefix, _, _, _, _ = template_fixture

    def forbidden(*args):
        raise AssertionError("No download should start")

    monkeypatch.setattr(http.app.state.services.dataset_templates.downloads, "fetch", forbidden)
    for body in [
        {"template_id": "openslide-cmu-small"},
        {"template_id": "Task09_Spleen", "section": "test"},
    ]:
        response = http.post(prefix + "/dataset-imports", json=body | {"split": "validation"})
        assert response.status_code == 422, response.text
    assert client.get(prefix + "/jobs") == []


def test_chat_combined_import_completes_both_portions_and_preserves_existing_evaluation(
    client, http, template_fixture
):
    from uuid import uuid4

    prefix, _, entries, write, mask = template_fixture
    for index in range(3, 42):
        entries[f"Task09_Spleen/imagesTr/spleen_{index}.nii.gz"] = nifti(
            np.full(mask.shape, index, dtype=np.int16)
        )
        entries[f"Task09_Spleen/labelsTr/spleen_{index}.nii.gz"] = nifti(mask)
    write(entries)
    existing = imported(client, prefix, split="validation", limit=5)
    before = client.get(prefix + "/assets")
    http.app.state.services.assistants.provider.queue = [
        chat_tool("inspect_workspace", collection="dataset_templates"),
        chat_tool("import_dataset_split", template_id="Task09_Spleen", evaluation_percentage=20),
    ]
    body = {
        "message": "From Medical Decathlon Speen Dataset import 80% images only for annotation "
        "and remaining 20% images+labels for evaluation.",
        "request_id": uuid4().hex,
    }
    reply = client.post(prefix + "/assistant", body)
    assert reply["tools"] == ["inspect_workspace", "import_dataset_split"]
    assert "80% images only" in reply["message"]
    assert "20% images with labels" in reply["message"]
    job = client.get("/api/jobs/" + reply["job_id"])
    assert job["request"]["limit"] is None
    result = client.wait(job["id"])
    assert not result["failed"]
    assert len(result["asset_ids"]) == 41
    assert len(result["annotation_asset_ids"]) == 32
    assert len(result["evaluation_asset_ids"]) == len(result["annotation_ids"]) == 9
    assert set(result["annotation_asset_ids"]).isdisjoint(result["evaluation_asset_ids"])
    assert set(existing["asset_ids"]) <= set(result["evaluation_asset_ids"])
    assets = client.get(prefix + "/assets")
    assert len(assets) == 41
    assert all(a in assets for a in before)
    assert all(a["annotation_id"] is None for a in assets if a["split"] == "pool")
    assert all(a["annotation_id"] for a in assets if a["split"] == "validation")
    assert client.get(prefix + "/decisions") == []
    record = client.get(prefix + "/evaluation-sets")[0]
    assert result["evaluation_set_id"] == record["id"] == existing["evaluation_set_id"]
    assert len(record["member_groups"]) == 9
    # HTTP retries and a fresh import both preserve identifiers, masks and reservations.
    body["conversation_id"] = reply["conversation_id"]
    assert client.post(prefix + "/assistant", body) == reply
    again = imported(client, prefix, evaluation_percentage=20, limit=None)
    assert again == result
    assert client.get(prefix + "/assets") == assets


def test_combined_import_honors_total_limit_offset_and_annotation_label_choice(
    client, template_fixture
):
    prefix, _, entries, write, mask = template_fixture
    for index in range(3, 8):
        entries[f"Task09_Spleen/imagesTr/spleen_{index}.nii.gz"] = nifti(
            np.full(mask.shape, index, dtype=np.int16)
        )
        entries[f"Task09_Spleen/labelsTr/spleen_{index}.nii.gz"] = nifti(mask)
    write(entries)
    result = imported(
        client, prefix, evaluation_percentage=20, offset=2, limit=5, include_masks=True
    )
    assert not result["failed"]
    assert len(result["asset_ids"]) == len(result["annotation_ids"]) == 5
    assert len(result["annotation_asset_ids"]) == 4
    assert len(result["evaluation_asset_ids"]) == 1
    assert {a["name"] for a in client.get(prefix + "/assets")} == {
        f"spleen_{i}.nii.gz" for i in range(3, 8)
    }


@pytest.mark.parametrize(
    "choices",
    [
        {"evaluation_percentage": 0},
        {"evaluation_percentage": 100},
        {"section": "test"},
        {"split": "validation"},
        {"template_id": "openslide-cmu-small"},
        {"template_id": "totalsegmentator-small"},
    ],
)
def test_combined_import_rejects_invalid_choices_before_download(
    http, client, template_fixture, monkeypatch, choices
):
    prefix, _, _, _, _ = template_fixture

    def forbidden(*args):
        raise AssertionError("Must not download")

    monkeypatch.setattr(http.app.state.services.dataset_templates.downloads, "fetch", forbidden)
    response = http.post(
        prefix + "/dataset-imports",
        json={"template_id": "Task09_Spleen", "evaluation_percentage": 20, **choices},
    )
    assert response.status_code == 422
    assert client.get(prefix + "/jobs") == []


def test_unused_evaluation_imports_can_be_deleted_then_reimported(client, http, template_fixture):
    from monailabel.core.evaluation import EvaluationReservation

    prefix, _, _, _, _ = template_fixture
    result = imported(client, prefix, split="validation", limit=None)
    assets = client.get(prefix + "/assets")
    record = client.get(prefix + "/evaluation-sets")[0]
    response = http.request("DELETE", prefix + "/assets", json={"asset_ids": [assets[0]["id"]]})
    assert response.status_code == 200, response.text
    current = client.get(prefix + "/evaluation-sets")[0]
    assert current["id"] == record["id"]
    assert current["version"] == record["version"] + 1
    assert current["member_groups"] == [assets[1]["group_id"]]
    assert (
        http.get("/api/annotations/" + assets[0]["annotation_id"] + "/mask.bin").status_code == 404
    )
    assert http.delete("/api/assets/" + assets[1]["id"]).status_code == 200
    assert client.get(prefix + "/assets") == []
    assert client.get(prefix + "/evaluation-sets")[0]["member_groups"] == []
    assert not http.app.state.services.store.list(EvaluationReservation, assets[0]["project_id"])
    retry = imported(client, prefix, evaluation_percentage=20, limit=None)
    assert not retry["failed"]
    assert len(retry["annotation_asset_ids"]) == len(retry["evaluation_asset_ids"]) == 1
    assert set(retry["asset_ids"]).isdisjoint(result["asset_ids"])


def test_saved_evaluation_references_still_block_atomic_deletion(client, http, template_fixture):
    prefix, _, _, _, _ = template_fixture
    result = imported(client, prefix, split="validation", limit=1)
    annotation = result["annotation_ids"][0]
    client.post("/api/annotations/" + annotation + "/decision", {"verdict": "accepted"})
    record = client.get(prefix + "/evaluation-sets")[0]
    version = client.post(
        prefix + "/evaluation-sets/" + record["id"] + "/versions",
        {"base_version": record["version"], "label_ids": [0, 4]},
    )
    other = imported(client, prefix, offset=1, limit=1)
    before = client.get(prefix + "/assets")
    response = http.request(
        "DELETE",
        prefix + "/assets",
        json={"asset_ids": result["asset_ids"] + other["asset_ids"]},
    )
    assert response.status_code == 409
    assert "saved evaluation references" in response.json()["detail"]
    assert client.get(prefix + "/assets") == before
    assert client.get(prefix + "/evaluation-set-versions") == [version]


def test_deleting_unused_evaluation_alias_keeps_remaining_case_reserved(
    client, http, template_fixture
):
    from monailabel.core.evaluation import EvaluationReservation

    prefix, _, entries, _, _ = template_fixture
    result = imported(client, prefix, split="validation", limit=1)
    asset = client.get(prefix + "/assets")[0]
    alias = http.post(
        prefix + "/assets/upload",
        params={"name": "alias.nii.gz", "group_id": asset["group_id"]},
        content=entries["Task09_Spleen/imagesTr/spleen_1.nii.gz"],
    ).json()
    assert alias["split"] == "validation"
    assert http.delete("/api/assets/" + asset["id"]).status_code == 200
    record = client.get(prefix + "/evaluation-sets")[0]
    assert record["id"] == result["evaluation_set_id"]
    assert record["member_groups"] == [alias["group_id"]]
    assert http.app.state.services.store.list(EvaluationReservation, asset["project_id"])
    assert http.post("/api/assets/" + alias["id"] + "/assign-train").status_code == 409
    assert http.delete("/api/assets/" + alias["id"]).status_code == 200
    assert not http.app.state.services.store.list(EvaluationReservation, asset["project_id"])
