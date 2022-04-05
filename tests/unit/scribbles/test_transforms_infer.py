# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np
from monai.utils import set_determinism
from parameterized import parameterized

from monailabel.scribbles.transforms import (
    AddBackgroundScribblesFromROId,
    InteractiveSegmentationTransform,
    MakeISegUnaryd,
    MakeLikelihoodFromScribblesHistogramd,
    SoftenProbSoftmax,
    WriteLogits,
)

set_determinism(seed=123)


def generate_synthetic_binary_segmentation(height, width, num_circles=10, r_min=10, r_max=100, random_state=None):
    # function based on:
    # https://github.com/Project-MONAI/MONAI/blob/dev/monai/data/synthetic.py

    if r_min > r_max:
        raise ValueError("r_min cannot be greater than r_max")

    min_size = min(height, width)
    if 2 * r_max > min_size:
        raise ValueError("2 * r_max cannot be greater than min side")

    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state

    mask = np.zeros((height, width), dtype=bool)
    for _ in range(num_circles):
        x = rs.randint(r_max, width - r_max)
        y = rs.randint(r_max, height - r_max)
        r = rs.randint(r_min, r_max)
        spy, spx = np.ogrid[-x : width - x, -y : height - y]
        circle = (spx * spx + spy * spy) <= r * r

        mask[circle] = True

    return mask


def add_salt_and_pepper_noise(data, p=0.05):
    if p <= 0:
        return data

    original_dtype = data.dtype

    random_image_data = np.random.choice([0, 1], p=[p, 1 - p], size=data.shape)

    return (data.astype(np.float32) * random_image_data).astype(original_dtype)


def generate_label_with_noise(height, width, label_key="label", noisy_key="noisy", pred_key="pred", num_slices=1):
    label = generate_synthetic_binary_segmentation(height, width, num_circles=10, r_min=10, r_max=50)
    noisy_invert = ~add_salt_and_pepper_noise(
        generate_synthetic_binary_segmentation(height, width, num_circles=10, r_min=10, r_max=50), p=0.7
    )
    noisy = label & noisy_invert
    label = np.expand_dims(label, axis=0).astype(np.float32)
    noisy = np.expand_dims(noisy, axis=0).astype(np.float32)

    if num_slices >= 1:
        if num_slices != 1:
            label = np.expand_dims(label, axis=0)
            noisy = np.expand_dims(noisy, axis=0)

            tmp_list = []
            for _ in range(num_slices):
                tmp_list.append(label)
            label = np.concatenate(tmp_list, axis=1)

            tmp_list = []
            for _ in range(num_slices):
                tmp_list.append(noisy)
            noisy = np.concatenate(tmp_list, axis=1)
    else:
        raise ValueError("unrecognised num_slices selected [{}]".format(num_slices))

    pred = label
    label = np.concatenate([1 - label, label], axis=0)

    return {label_key: label, noisy_key: noisy, pred_key: pred}


HEIGHT = 128
WIDTH = 128
NUM_SLICES = 32

# generate 2d noisy data
two_dim_data = generate_label_with_noise(
    height=HEIGHT, width=WIDTH, label_key="prob", noisy_key="image", pred_key="target", num_slices=1
)
# generate 3d noisy data
three_dim_data = generate_label_with_noise(
    height=HEIGHT, width=WIDTH, label_key="prob", noisy_key="image", pred_key="target", num_slices=NUM_SLICES
)


TEST_CASE_OPTIM_TX = [
    # 2D case
    (
        {"unary": "prob", "pairwise": "image"},
        {"prob": two_dim_data["prob"], "image": two_dim_data["image"]},
        {"target": two_dim_data["target"]},
        (1, HEIGHT, WIDTH),
    ),
    # 3D case
    (
        {"unary": "prob", "pairwise": "image"},
        {"prob": three_dim_data["prob"], "image": three_dim_data["image"]},
        {"target": three_dim_data["target"]},
        (1, NUM_SLICES, HEIGHT, WIDTH),
    ),
]

TEST_CASE_ISEG_OPTIM_TX = [
    # 2D case
    (
        {
            "image": "image",
            "logits": "prob",
            "scribbles": "scribbles",
            "scribbles_bg_label": 2,
            "scribbles_fg_label": 3,
        },
        {"image": two_dim_data["image"], "prob": two_dim_data["prob"], "scribbles": two_dim_data["prob"][[1], ...] + 2},
        {"target": two_dim_data["target"]},
        (1, HEIGHT, WIDTH),
    ),
    # 3D case
    (
        {
            "image": "image",
            "logits": "prob",
            "scribbles": "scribbles",
            "scribbles_bg_label": 2,
            "scribbles_fg_label": 3,
        },
        {
            "image": three_dim_data["image"],
            "prob": three_dim_data["prob"],
            "scribbles": three_dim_data["prob"][[1], ...] + 2,
        },
        {"target": three_dim_data["target"]},
        (1, NUM_SLICES, HEIGHT, WIDTH),
    ),
]

TEST_CASE_MAKE_ISEG_UNARY_TX = [
    # 2D case
    (
        {
            "image": "image",
            "logits": "prob",
            "scribbles": "scribbles",
            "scribbles_bg_label": 2,
            "scribbles_fg_label": 3,
        },
        {"image": two_dim_data["image"], "prob": two_dim_data["prob"], "scribbles": two_dim_data["prob"][[1], ...] + 2},
        {"target": two_dim_data["prob"]},
        (2, HEIGHT, WIDTH),
    ),
    # 3D case
    (
        {
            "image": "image",
            "logits": "prob",
            "scribbles": "scribbles",
            "scribbles_bg_label": 2,
            "scribbles_fg_label": 3,
        },
        {
            "image": three_dim_data["image"],
            "prob": three_dim_data["prob"],
            "scribbles": three_dim_data["prob"][[1], ...] + 2,
        },
        {"target": three_dim_data["prob"]},
        (2, NUM_SLICES, HEIGHT, WIDTH),
    ),
]

TEST_CASE_MAKE_LIKE_HIST_TX = [
    # 2D case
    (
        {"image": "image", "scribbles": "scribbles", "scribbles_bg_label": 2, "scribbles_fg_label": 3},
        {"image": two_dim_data["target"], "scribbles": two_dim_data["prob"][[1], ...] + 2},
        {"target": two_dim_data["prob"]},
        (2, HEIGHT, WIDTH),
    ),
    # 3D case
    (
        {"image": "image", "scribbles": "scribbles", "scribbles_bg_label": 2, "scribbles_fg_label": 3},
        {"image": three_dim_data["target"], "scribbles": three_dim_data["prob"][[1], ...] + 2},
        {"target": three_dim_data["prob"]},
        (2, NUM_SLICES, HEIGHT, WIDTH),
    ),
]


TEST_CASE_ADD_BG_ROI = [
    (
        {"scribbles": "scribbles", "roi_key": "roi", "scribbles_bg_label": 2, "scribbles_fg_label": 3},
        {
            "scribbles": np.zeros((1, NUM_SLICES, HEIGHT, WIDTH), dtype=np.float32),
            "roi": [
                NUM_SLICES // 2 - 4,
                NUM_SLICES // 2 + 4,
                HEIGHT // 2 - 8,
                HEIGHT // 2 + 8,
                WIDTH // 2 - 16,
                WIDTH // 2 + 16,
            ],
        },
        (1, NUM_SLICES, HEIGHT, WIDTH),
    ),
]

TEST_CASE_HISTOGRAM_GRAPHCUT = [
    # 2D case
    (
        {
            "image": np.squeeze(two_dim_data["image"]),
            "label": np.squeeze(two_dim_data["prob"][[1], ...] + 2),
            "image_meta_dict": {"affine": np.identity(4)},
            "label_meta_dict": {"affine": np.identity(4)},
        },
        (1, HEIGHT, WIDTH),
    ),
    # 3D case
    (
        {
            "image": np.squeeze(three_dim_data["image"]),
            "label": np.squeeze(three_dim_data["prob"][[1], ...] + 2),
            "image_meta_dict": {"affine": np.identity(5)},
            "label_meta_dict": {"affine": np.identity(5)},
        },
        (NUM_SLICES, HEIGHT, WIDTH),
    ),
]


class TestScribblesTransforms(unittest.TestCase):
    @parameterized.expand(TEST_CASE_ADD_BG_ROI)
    def test_add_bg_roi_transform(self, input_param, test_input, expected_shape):
        # float32 test
        result = AddBackgroundScribblesFromROId(**input_param)(test_input)
        input_data = test_input["scribbles"].copy()
        mask = input_data.astype(bool)
        mask[
            :,
            test_input["roi"][0] : test_input["roi"][1],
            test_input["roi"][2] : test_input["roi"][3],
            test_input["roi"][4] : test_input["roi"][5],
        ] = True
        input_data[~mask] = input_param["scribbles_bg_label"]

        np.testing.assert_equal(input_data, result["scribbles"])
        self.assertTupleEqual(expected_shape, result["scribbles"].shape)
        self.assertTupleEqual(test_input["scribbles"].shape, result["scribbles"].shape)

        # int32 test
        test_input["scribbles"] = test_input["scribbles"].astype(np.int)
        result = AddBackgroundScribblesFromROId(**input_param)(test_input)
        input_data = test_input["scribbles"].copy()
        mask = input_data.astype(bool)
        mask[
            :,
            test_input["roi"][0] : test_input["roi"][1],
            test_input["roi"][2] : test_input["roi"][3],
            test_input["roi"][4] : test_input["roi"][5],
        ] = True
        input_data[~mask] = input_param["scribbles_bg_label"]

        np.testing.assert_equal(input_data, result["scribbles"])
        self.assertTupleEqual(expected_shape, result["scribbles"].shape)
        self.assertTupleEqual(test_input["scribbles"].shape, result["scribbles"].shape)

    @parameterized.expand(TEST_CASE_MAKE_ISEG_UNARY_TX)
    def test_make_iseg_unary_transform(self, input_param, test_input, output, expected_shape):
        input_param.update({"unary": "pred"})
        result = MakeISegUnaryd(**input_param)(test_input)

        # make expected unary output
        expected_result = output["target"].copy()
        eps = np.finfo(expected_result.dtype).eps
        expected_result[expected_result == 0] = eps
        expected_result[expected_result == 1] = 1 - eps

        # compare
        np.testing.assert_equal(expected_result, result["pred"])
        self.assertTupleEqual(expected_shape, result["pred"].shape)

        with self.assertRaises(ValueError):
            test_input["prob"] = np.random.rand(3, 128, 128, 128)
            result = MakeISegUnaryd(**input_param)(test_input)

    @parameterized.expand(TEST_CASE_MAKE_LIKE_HIST_TX)
    def test_make_likelihood_histogram(self, input_param, test_input, output, expected_shape):
        input_param.update({"post_proc_label": "pred"})
        result = MakeLikelihoodFromScribblesHistogramd(**input_param)(test_input)

        # make expected output
        expected_result = np.argmax(output["target"].copy(), axis=0)

        # compare
        np.testing.assert_equal(expected_result, np.argmax(result["pred"], axis=0))
        self.assertTupleEqual(expected_shape, result["pred"].shape)

    @parameterized.expand(TEST_CASE_ISEG_OPTIM_TX)
    def test_writelogits(self, input_param, test_input, output, expected_shape):
        test_input.update({"image_path": "./image.nii.gz"})
        result = WriteLogits(key="image")(test_input)
        self.assertEqual(os.path.exists(result["result"]["image"]), True)

    def test_interactive_seg_transforms(self):
        class MyInteractiveSeg(InteractiveSegmentationTransform):
            def __init__(self, meta_key_postfix):
                super().__init__(meta_key_postfix)

            def __call__(self, data):
                return data

        iseg_tx = MyInteractiveSeg(meta_key_postfix="meta_dict")
        data = {"image": [0, 1, 2, 3, 4, 5]}
        self.assertEqual(iseg_tx._fetch_data(data, "image"), data["image"])

        image_np = np.random.rand(2, 128, 128)
        outimage_np = iseg_tx._normalise_logits(image_np, axis=0)
        self.assertEqual(np.sum(outimage_np, axis=0).mean(), 1.0)

        data.update({"image_meta_dict": {"affine": [0, 1, 2, 3, 4, 5]}})
        data = iseg_tx._copy_affine(data, "image", "label")
        self.assertIn("label_meta_dict", data.keys())

    def test_soften_prob_softmax_transforms(self):
        soften_tx = SoftenProbSoftmax(logits="logits", meta_key_postfix="meta_dict", prob="prob")
        data = {"logits": np.random.rand(2, 128, 128, 128)}
        output_data = soften_tx(data)

        # minimum should be close to 0,1
        self.assertAlmostEqual(round(output_data["prob"].min(), 1), 0.1)
        # maximum should be close to 0.9
        self.assertAlmostEqual(round(output_data["prob"].max(), 1), 0.9)

        # shape should be same as input
        self.assertEqual(output_data["prob"].shape, data["logits"].shape)


if __name__ == "__main__":
    unittest.main()
