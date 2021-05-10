import os

import numpy as np


from monai.engines.workflow import Engine
from monai.utils import optional_import

nib, _ = optional_import("nibabel")
torchvision, _ = optional_import("torchvision")
make_grid, _ = optional_import("torchvision.utils", name="make_grid")
Image, _ = optional_import("PIL.Image")
ImageDraw, _ = optional_import("PIL.ImageDraw")

from inner_event.Events_deep_edit import DeepEditEvents

class InnerIterSaver:
    def __init__(
        self,
        output_dir: str = "./runs",
        images=True,
    ):
        self.output_dir = output_dir
        self.images = images
        os.makedirs(self.output_dir, exist_ok=True)

    def attach(self, engine: Engine) -> None:
        if not engine.has_event_handler(self, DeepEditEvents.INNER_ITERATION_COMPLETED):
            engine.add_event_handler(DeepEditEvents.INNER_ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine):
        batch_data = engine.state.batch_data_inner
        preds_data = engine.state.preds_inner
        tag = "inner_iter_"
        for bidx in range(len(batch_data.get("image"))):
            step = engine.state.iteration
            step_inner = engine.state.step_inner

            image = batch_data["image"][bidx][0].detach().cpu().numpy()[np.newaxis]
            label = batch_data["label"][bidx].detach().cpu().numpy()
            pred = preds_data[bidx].detach().cpu().numpy()

            if self.images and len(image.shape) == 4:
                samples = {"image": batch_data["image"][bidx].detach().cpu().numpy(), "label": label[0], "pred": pred[0]}
                print('Saving image, label and pred for inner iteration: ', step_inner)
                for sample in samples:
                    if sample != 'image':
                        img = samples[sample] # img = np.moveaxis(samples[sample], -3, -1)
                    else:
                        img = np.moveaxis(samples[sample], -4, -1)
                    img = nib.Nifti1Image(img, np.eye(4))
                    nib.save(
                        img,
                        os.path.join(
                            self.output_dir, "{}{}_{:0>4d}_{:0>2d}_inner_iter_{:0>2d}.nii.gz".format(tag, sample, step, step_inner, bidx)
                        ),
                    )
