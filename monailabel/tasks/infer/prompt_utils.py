# custom_utils.py

import os
import shutil
import torch
import numpy as np
from monai.utils import optional_import
from monai.data import decollate_batch
from typing import Any, Dict
from einops import rearrange

rearrange, _ = optional_import("einops", name="rearrange")


def transform_points(point, affine):
    bs, n = point.shape[:2]
    point = np.concatenate((point, np.ones((bs, n, 1))), axis=-1)
    point = rearrange(point, "b n d -> d (b n)")
    point = affine @ point
    point = rearrange(point, "d (b n)-> b n d", b=bs)[:, :, :3]
    return point

def check_prompts_format(label_prompt, points, point_labels):
    """check the format of user prompts
    label_prompt: [1,2,3,4,...,B] List of tensors
    points: [[[x,y,z], [x,y,z], ...]] List of coordinates of a single object
    point_labels: [[1,1,0,...]] List of scalar that matches number of points
    """
    # check prompt is given
    if label_prompt is None and points is None:
        everything_labels = list(set([i+1 for i in range(132)]) - set([2,16,18,20,21,23,24,25,26,27,128,129,130,131,132]))
        if everything_labels is not None:
            label_prompt = [torch.tensor(_) for _ in everything_labels]

            return label_prompt, points, point_labels
        else:
            raise ValueError("Prompt must be given for inference.")
    # check label_prompt
    if label_prompt is not None:

        if isinstance(label_prompt, list):
            # if not np.all([len(_) == 1 for _ in label_prompt]):
            #     raise ValueError("Label prompt must be a list of single scalar, [1,2,3,4,...,].")
            if not np.all([(x < 255).item() for x in label_prompt]):
                raise ValueError("Current bundle only supports label prompt smaller than 255.")
            if points is None:
                supported_list = list({i + 1 for i in range(132)} - {16, 18, 129, 130, 131})
                if not np.all([x in supported_list for x in label_prompt]):
                    raise ValueError("Undefined label prompt detected. Provide point prompts for zero-shot.")
        else:
            raise ValueError("Label prompt must be a list, [1,2,3,4,...,].")
    # check points
    if points is not None:
        if point_labels is None:
            raise ValueError("Point labels must be given if points are given.")
        if not np.all([len(_) == 3 for _ in points]):
            raise ValueError("Points must be three dimensional (x,y,z) in the shape of [[x,y,z],...,[x,y,z]].")
        if len(points) != len(point_labels):
            raise ValueError("Points must match point labels.")
        if not np.all([_ in [-1, 0, 1, 2, 3] for _ in point_labels]):
            raise ValueError("Point labels can only be -1,0,1 and 2,3 for special flags.")
    if label_prompt is not None and points is not None:
        if len(label_prompt) != 1:
            raise ValueError("Label prompt can only be a single object if provided with point prompts.")
    # check point_labels
    if point_labels is not None:
        if points is None:
            raise ValueError("Points must be given if point labels are given.")
    return label_prompt, points, point_labels

def prompt_run_inferer(data: Dict[str, Any], inferer, network, input_key="image", output_label_key="pred", device="cuda", convert_to_batch=True):
    # Retrieve label_prompt, points, and point_labels
    label_prompt, points, point_labels = (
        data.get("label_prompt", None),
        data.get("points", None),
        data.get("point_labels", None),
    )

    if label_prompt is not None:
        label_prompt = [torch.tensor(_) for _ in label_prompt]
    if isinstance(label_prompt, torch.Tensor):
        if label_prompt.numel() == 0:
            label_prompt = None
    elif isinstance(label_prompt, list):
        if len(label_prompt) == 0:
            label_prompt = None

    label_prompt, points, point_labels = check_prompts_format(label_prompt, points, point_labels)
    label_prompt = (
        torch.as_tensor([label_prompt]).to(torch.device(device))[0].unsqueeze(-1) if label_prompt is not None else None
    )
    data["label_prompt"] = label_prompt

    # Transform points based on spatial scaling factors
    if points is not None:
        points = torch.as_tensor([points])

        original_spatial_shape = np.array(data['image_meta_dict']['spatial_shape'])
        resized_spatial_shape = np.array(data[input_key].shape[1:])
        scaling_factors = resized_spatial_shape / original_spatial_shape
        transformed_point = points * scaling_factors
        transformed_point_rounded = np.round(transformed_point)
        points = transformed_point_rounded.to(torch.device(device))

    point_labels = torch.as_tensor([point_labels]).to(torch.device(device)) if point_labels is not None else None
    data["points"] = points
    data["point_labels"] = point_labels

    inputs = data[input_key]
    inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
    inputs = inputs[None].to(torch.device(device))
    inputs = inputs.to(torch.device(device))


    with torch.no_grad():
        outputs = inferer(inputs, network, point_coords=points, point_labels=point_labels, class_vector=label_prompt)


    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if convert_to_batch:
        if isinstance(outputs, dict):
            outputs_d = decollate_batch(outputs)
            outputs = outputs_d[0]
        else:
            outputs = outputs[0]

    data[output_label_key] = outputs[0] if isinstance(outputs, list) else outputs
    return data
