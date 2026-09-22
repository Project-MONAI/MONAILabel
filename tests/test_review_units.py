"""Independent review coverage and immutable history within one source file."""

import copy

import numpy as np
import test_image_regions
import test_videos

from monailabel.core.models import Annotation, ModelRecord, Project, ReviewDecision, Snapshot
from monailabel.core.ports import IGNORE_LABEL
from monailabel.core.review_units import UnitAnnotation
from monailabel.core.video import TrackDocument
from monailabel.server.learning_data.arrays import SampleArrays, polygon_mask
from monailabel.server.review_units.service import ReviewUnits

region_setup = test_image_regions.region_setup
clip = test_videos.clip
video = test_videos.video


def unit_list(client, project_id):
    return client.get(f"/api/projects/{project_id}/review-units")


def accept(client, unit):
    return client.post(
        f"/api/review-units/{unit['id']}/decision",
        {
            "base_revision": unit["revision"],
            "verdict": "accepted",
        },
    )


def region_submit(client, asset, revision, regions, mask):
    return client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": revision,
            "regions": regions,
            "covered_labels": [0, 1, 2],
            "mask": mask.tolist(),
        },
    )


def test_regions_preserve_unrelated_acceptance_and_source_pixels(client, http, region_setup):
    project, asset, _, _ = region_setup
    first = {"x": 1, "y": 1, "width": 4, "height": 4}
    second = {"x": 8, "y": 5, "width": 4, "height": 4}
    draft = np.ones((12, 16), dtype=np.uint8)
    initial = region_submit(client, asset, 0, [first], draft)
    units = unit_list(client, project["id"])
    assert len(units) == 1 and units[0]["name"] == "Region 1"
    accepted = accept(client, units[0])
    services = http.app.state.services
    saved = services.artifacts.array(services.store.get(Annotation, initial["id"]).mask_key)
    assert np.count_nonzero(saved) == 16  # Unsubmitted pixels are not persisted as labels.
    region_submit(client, asset, 1, [second], draft)
    revised_units = unit_list(client, project["id"])
    assert len(revised_units) == 2
    assert revised_units[0] == units[0]
    assert client.get(f"/api/projects/{project['id']}/decisions") == [accepted]
    draft[2, 2] = 2
    region_submit(client, asset, 2, [first], draft)
    revised = unit_list(client, project["id"])
    assert revised[0]["revision"] == 2
    assert revised[0]["annotation_id"] != units[0]["annotation_id"]
    assert revised[1] == revised_units[1]
    old = services.store.get(UnitAnnotation, units[0]["annotation_id"])
    assert services.artifacts.array(old.mask_key)[2, 2] == 1


def test_overlapping_region_submission_rolls_back(client, http, region_setup):
    project, asset, _, _ = region_setup
    first = {"x": 1, "y": 1, "width": 4, "height": 4}
    draft = np.ones((12, 16), dtype=np.uint8)
    region_submit(client, asset, 0, [first], draft)
    response = http.post(
        f"/api/assets/{asset['id']}/review",
        json={
            "base_revision": 1,
            "regions": [{**first, "x": 2}],
            "covered_labels": [0, 1],
            "mask": draft.tolist(),
        },
    )
    assert response.status_code == 409 and "overlaps Region 1" in response.text
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 1
    assert len(unit_list(client, project["id"])) == 1


def test_video_ranges_review_independently_and_boxes_are_not_masks(client, http, video):
    def track(identifier, start, stop):
        return {
            "id": identifier,
            "label_id": 1,
            "keyframes": [
                {"frame": n, "points": [1, 1, 20, 1, 20, 20, 1, 20]} for n in range(start, stop)
            ]
            + [{"frame": stop, "points": [1, 1, 20, 1, 20, 20, 1, 20], "outside": True}],
        }

    doc = {"tracks": [track("first", 0, 2)]}
    client.post(f"/api/videos/{video['id']}/review", {"base_revision": 0, "document": doc})
    first = unit_list(client, video["project_id"])[0]
    assert first["scope"] == {"kind": "frames", "start": 0, "stop": 2}
    accept(client, first)
    doc["tracks"].append(track("second", 3, 5))
    client.post(f"/api/videos/{video['id']}/review", {"base_revision": 1, "document": doc})
    units = unit_list(client, video["project_id"])
    assert units[0] == first and len(units) == 2
    changed = copy.deepcopy(doc)
    changed["tracks"][1]["keyframes"][0]["points"][2] = 21
    client.post(f"/api/videos/{video['id']}/review", {"base_revision": 2, "document": changed})
    after = unit_list(client, video["project_id"])
    assert after[0] == first and after[1]["revision"] == 2
    changed["tracks"].append(
        {"id": "box", "label_id": 2, "keyframes": [{"frame": 5, "box": [1, 1, 20, 20]}]}
    )
    client.post(f"/api/videos/{video['id']}/review", {"base_revision": 3, "document": changed})
    box = unit_list(client, video["project_id"])[-1]
    annotation = http.app.state.services.store.get(UnitAnnotation, box["annotation_id"])
    assert annotation.covered_labels == [0]


def train_small_unet(client, http, project_id):
    prefix = f"/api/projects/{project_id}"
    learner = client.post(
        prefix + "/learners",
        {
            "name": "Local specialist",
            "recipe": "monai-unet",
            "label_ids": [0, 1],
            "config": {
                "epochs": 1,
                "steps_per_epoch": 2,
                "patch_size": 16,
                "channels": [4, 8, 16, 32],
                "device": "cpu",
            },
        },
    )
    job = client.post(prefix + f"/learners/{learner['id']}/train", {})
    result = client.wait(job["id"])
    service = http.app.state.services
    snapshot = service.store.get(Snapshot, result["snapshot_id"])
    model = service.store.get(ModelRecord, result["model_id"])
    assert not snapshot.evaluation_requested and all(s.split == "train" for s in snapshot.samples)
    report = client.get(f"/api/jobs/{job['id']}/training-report")
    assert report["metrics"] is None and report["error"] is None
    assert report["final_loss"] is not None
    with SampleArrays(service.artifacts, snapshot.samples, [0, 1], lambda: None) as arrays:
        image, mask = arrays[0]
        output = service.models.predict(
            service.store.get(Project, project_id), model, image, "Segment the target", None
        )
        assert output.shape == mask.shape and set(np.unique(output)) <= {0, 1}
        return snapshot, mask


def test_accepted_pathology_region_trains_unet_and_excludes_unreviewed_pixels(
    client, http, region_setup
):
    project, asset, _, _ = region_setup
    first = {"x": 1, "y": 1, "width": 4, "height": 4, "runs": [[0, 10]]}
    draft = np.zeros((12, 16), dtype=np.uint8)
    draft[2, 2:4] = 1
    region_submit(client, asset, 0, [first], draft)
    unit = unit_list(client, project["id"])[0]
    accept(client, unit)
    region_submit(client, asset, 1, [{"x": 8, "y": 5, "width": 4, "height": 4}], draft)
    snapshot, mask = train_small_unet(client, http, project["id"])
    assert len(snapshot.samples) == 1 and snapshot.samples[0].unit_id == unit["id"]
    assert np.count_nonzero(mask == IGNORE_LABEL) == 6
    assert snapshot.samples[0].image_key == asset["image_key"]


def test_accepted_video_polygons_train_unet_from_original_frames(client, http, video):
    document = {
        "tracks": [
            {
                "id": "tool",
                "label_id": 1,
                "keyframes": [
                    {"frame": n, "points": [5, 5, 20, 5, 20, 20, 5, 20]} for n in range(2)
                ]
                + [{"frame": 2, "points": [5, 5, 20, 5, 20, 20, 5, 20], "outside": True}],
            }
        ]
    }
    client.post(f"/api/videos/{video['id']}/review", {"base_revision": 0, "document": document})
    unit = unit_list(client, video["project_id"])[0]
    accept(client, unit)
    snapshot, mask = train_small_unet(client, http, video["project_id"])
    assert len(snapshot.samples) == 2
    assert [s.video_frame.index for s in snapshot.samples] == [0, 1]
    assert {s.image_key for s in snapshot.samples} == {video["source_key"]}
    assert {s.group_id for s in snapshot.samples} == {video["group_id"]}
    assert mask.shape == (video["height"], video["width"])
    assert mask[10, 10] == 1 and mask[0, 0] == 0


def test_video_migration_preserves_existing_review_and_is_idempotent(client, http, video):
    annotation = client.post(
        f"/api/videos/{video['id']}/review",
        {"base_revision": 0, "document": test_videos.document()},
    )
    client.post(f"/api/videos/{video['id']}/decision", {"base_revision": 1, "verdict": "accepted"})
    service = http.app.state.services
    # Reproduce a workspace saved before scoped video review was introduced.
    with service.store.transaction() as session:
        session.connection.execute(
            "DELETE FROM records WHERE kind IN ('ReviewUnit','UnitAnnotation')"
        )
        session.connection.execute(
            "DELETE FROM records WHERE kind='ReviewDecision' AND "
            "json_extract(data, '$.annotation_id') != ?",
            (annotation["id"],),
        )
    reviews = ReviewUnits(service.store, service.artifacts)
    reviews.migrate_videos()
    units = unit_list(client, video["project_id"])
    assert len(units) == 2  # Frames 0–3 and 5, preserving the outside frame at 4.
    decisions = service.store.list(ReviewDecision, video["project_id"])
    assert len(decisions) == 3 and all(d.verdict == "accepted" for d in decisions)
    reviews.migrate_videos()
    assert unit_list(client, video["project_id"]) == units
    assert service.store.list(ReviewDecision, video["project_id"]) == decisions


def test_video_boxes_do_not_become_false_background_training_labels():
    document = TrackDocument.model_validate(
        {
            "tracks": [
                {"id": "box", "label_id": 1, "keyframes": [{"frame": 0, "box": [2, 2, 12, 12]}]},
                {
                    "id": "outline",
                    "label_id": 1,
                    "keyframes": [{"frame": 0, "points": [4, 4, 8, 4, 8, 8, 4, 8]}],
                },
            ]
        }
    )
    mask = polygon_mask(document, 0, 16, 16)
    assert mask[0, 0] == 0
    assert mask[3, 3] == IGNORE_LABEL
    assert mask[6, 6] == 1
