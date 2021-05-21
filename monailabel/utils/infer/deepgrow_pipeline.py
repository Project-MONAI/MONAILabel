import logging
import os

import numpy as np
import torch
from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    AddInitialSeedPointd,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropGuidanced,
)
from monai.inferers import SimpleInferer
from monai.transforms import (
    AddChanneld,
    AsChannelFirst,
    AsChannelFirstd,
    AsChannelLastd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    Spacingd,
    ToTensor,
)

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import BoundingBoxd, LargestCCd

logger = logging.getLogger(__name__)


class InferDeepgrowPipeline(InferTask):
    def __init__(
        self,
        path,
        model_3d: InferTask,
        network=None,
        type=InferType.DEEPGROW,
        dimension=3,
        description="Combines Deepgrow 2D model with any 3D segmentation/deepgrow model",
        spatial_size=(256, 256),
        model_size=(256, 256),
        batch_size=32,
        min_point_density=10,
        max_random_points=10,
        random_point_density=1000,
        output_largest_cc=False,
    ):
        super().__init__(
            path=path, network=network, type=type, labels=None, dimension=dimension, description=description
        )
        self.model_3d = model_3d
        self.spatial_size = spatial_size
        self.model_size = model_size

        self.batch_size = batch_size
        self.min_point_density = min_point_density
        self.max_random_points = max_random_points
        self.random_point_density = random_point_density
        self.output_largest_cc = output_largest_cc

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            AsChannelFirstd(keys="image"),
            Spacingd(keys="image", pixdim=[1.0, 1.0, 1.0], mode="bilinear"),
            AddGuidanceFromPointsd(ref_image="image", guidance="guidance", dimensions=3),
            AddChanneld(keys="image"),
            SpatialCropGuidanced(keys="image", guidance="guidance", spatial_size=self.spatial_size),
            Resized(keys="image", spatial_size=self.model_size, mode="area"),
            ResizeGuidanced(guidance="guidance", ref_image="image"),
            NormalizeIntensityd(keys="image", subtrahend=208, divisor=388),
            AddGuidanceSignald(image="image", guidance="guidance"),
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            LargestCCd(keys="pred"),
            RestoreLabeld(keys="pred", ref_image="image", mode="nearest"),
            AsChannelLastd(keys="pred"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]

    def __call__(self, request):
        result_file, result_json = self.model_3d(request)

        label = LoadImage(image_only=True)(result_file)
        label = AsChannelFirst()(label)
        logger.debug(f"Label shape: {label.shape}")

        foreground, slices = self.get_slices_points(label, request.get("foreground", []))
        if os.path.exists(result_file):
            os.unlink(result_file)

        request["foreground"] = foreground
        request["slices"] = slices

        # TODO:: fix multi-thread issue
        self.model_size = (label.shape[0], self.model_size[-2], self.model_size[-1])
        logger.info(f"Model Size: {self.model_size}")

        result_file, j = super().__call__(request)
        result_json.update(j)
        return result_file, result_json

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        image = data[self.input_key]
        slices = data["slices"]
        logger.debug("Pre processed Image shape: {}".format(image.shape))

        batched_data = []
        batched_slices = []
        pred = np.zeros(image.shape[1:])
        logger.debug("Init pred: {}".format(pred.shape))

        for slice_idx in slices:
            img = np.array([image[0][slice_idx], image[1][slice_idx], image[2][slice_idx]])
            # logger.info('{} => Image shape: {}'.format(slice_idx, img.shape))

            batched_data.append(img)
            batched_slices.append(slice_idx)
            if 0 < self.batch_size == len(batched_data):
                self.run_batch(super().run_inferer, batched_data, batched_slices, pred)
                batched_data = []
                batched_slices = []

        # Last batch
        if len(batched_data):
            self.run_batch(super().run_inferer, batched_data, batched_slices, pred)

        pred = pred[np.newaxis]
        logger.debug("Prediction: {}; sum: {}".format(pred.shape, np.sum(pred)))

        data[self.output_label_key] = pred
        return data

    def run_batch(self, run_inferer_method, batched_data, batched_slices, pred):
        to_tensor = ToTensor()

        bdata = {self.input_key: to_tensor(batched_data)}
        outputs = run_inferer_method(bdata, False)
        for i, s in enumerate(batched_slices):
            p = torch.sigmoid(outputs[self.output_label_key][i]).detach().cpu().numpy()
            p[p > 0.5] = 1
            pred[s] = LargestCCd.get_largest_cc(p) if self.output_largest_cc else p

    def get_random_points(self, label):
        points = []
        count = min(self.max_random_points, int(np.sum(label) // self.random_point_density))
        if count:
            label_idx = np.where(label > 0.5)
            for _ in range(count):
                seed = np.random.randint(0, len(label_idx[0]))
                points.append([label_idx[0][seed], label_idx[1][seed]])
        return points

    def get_slices_points(self, label, initial_foreground):
        logger.debug("Label shape: {}".format(label.shape))

        foreground_all = initial_foreground
        max_slices = label.shape[0]
        for i in range(max_slices):
            lab = label[i, :, :]
            if np.sum(lab) == 0:
                continue

            lab = lab[np.newaxis]
            foreground = []

            # get largest cc
            lab = LargestCCd.get_largest_cc(lab)
            if np.sum(lab) < self.min_point_density:
                logger.debug("Ignoring this slice: {}; min existing points: {}".format(i, self.min_point_density))
                continue

            # Add initial point  based on CDT/Distance
            t = AddInitialSeedPointd()
            guidance = t._apply(lab, None)
            for point in guidance[0]:
                if np.any(np.asarray(point) < 0):
                    continue
                foreground.append([point[-2], point[-1]])
                foreground_all.append([point[-2], point[-1], i])

            # Add Random points
            points = self.get_random_points(lab[0])
            for point in points:
                foreground.append([point[-2], point[-1]])
                foreground_all.append([point[-2], point[-1], i])
            # logger.debug('Slice: {}; Sum: {}; Foreground Points: {}'.format(i, np.sum(lab), foreground))

        logger.info("Total Foreground Points: {}".format(len(foreground_all)))
        slices = list(set((np.array(foreground_all)[:, 2]).astype(int).tolist()))
        logger.info("Total slices: {}; min: {}; max: {}".format(len(slices), min(slices), max(slices)))
        return foreground_all, slices
