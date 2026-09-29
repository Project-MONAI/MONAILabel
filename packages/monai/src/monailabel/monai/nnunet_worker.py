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

"""Private nnU-Net planning, training and prediction process.

Only training cases enter this process. MONAI Label evaluates the final
checkpoint separately; upstream cross-validation and checkpoint selection are
deliberately not run on the application's held-out references.
"""

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch

from monailabel.monai.nnunet_config import NNUNetConfig
from monailabel.monai.nnunet_runtime import DATASET


def status(root: Path, value: float, message: str) -> None:
    temporary = root / "progress.part"
    temporary.write_text(json.dumps({"progress": value, "message": message}))
    temporary.replace(root / "progress.json")


def fingerprint(runner: Any, root: Path, dataset: dict[str, Any]) -> None:
    # Upstream foreground statistics include every positive ID, including the
    # ignore ID. Hide unreviewed coverage only while collecting those statistics;
    # preprocessing and loss must receive the original ignore mask.
    labels = root / "raw" / DATASET / "labelsTr"
    original = labels.with_name("reviewed_labels")
    labels.rename(original)
    labels.mkdir()
    try:
        for path in original.glob("*.nii.gz"):
            source = cast(Any, nib.load(path))
            values = np.asarray(source.dataobj).copy()
            values[values == dataset["labels"]["ignore"]] = 0
            nib.save(nib.Nifti1Image(values, source.affine), labels / path.name)  # type: ignore[no-untyped-call]
        runner.extract_fingerprints(npfp=1, verify_dataset_integrity=True)
    finally:
        shutil.rmtree(labels)
        original.rename(labels)


def training_batches(trainer: Any) -> Any:
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    trainer.dataset_class = infer_dataset_class(trainer.preprocessed_dataset_folder)
    data = trainer.dataset_class(trainer.preprocessed_dataset_folder)
    rotation, dummy_2d, initial_patch, mirrors = (
        trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    )
    transforms = trainer.get_training_transforms(
        trainer.configuration_manager.patch_size,
        rotation,
        trainer._get_deep_supervision_scales(),
        mirrors,
        dummy_2d,
        use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm,
        is_cascaded=False,
        foreground_labels=trainer.label_manager.foreground_labels,
        ignore_label=trainer.label_manager.ignore_label,
    )
    return nnUNetDataLoader(
        data,
        trainer.batch_size,
        initial_patch,
        trainer.configuration_manager.patch_size,
        trainer.label_manager,
        oversample_foreground_percent=trainer.oversample_foreground_percent,
        transforms=transforms,
    )


def train(root: Path, request: dict[str, Any]) -> None:
    from monai.apps.nnunet.nnunetv2_runner import nnUNetV2Runner
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    config = NNUNetConfig.model_validate(request["config"])
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    runner = nnUNetV2Runner(
        {
            "nnunet_raw": str(root / "raw"),
            "nnunet_preprocessed": str(root / "preprocessed"),
            "nnunet_results": str(root / "results"),
            "dataset_name_or_id": 1,
        },
        work_dir=str(root),
    )
    prepared = root / "preprocessed" / DATASET
    model = root / "model"
    dataset = json.loads((root / "raw" / DATASET / "dataset.json").read_text())
    if request["mode"] == "scratch":
        status(root, 0.02, "Planning nnU-Net from training volumes.")
        fingerprint(runner, root, dataset)
        runner.plan_experiments(pl="nnUNetPlannerResEncL", gpu_memory_target=24)
    else:
        status(root, 0.02, "Reusing the parent nnU-Net plan and normalization.")
        prepared.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model / "plans.json", prepared / "nnUNetPlans.json")
        shutil.copyfile(model / "dataset_fingerprint.json", prepared / "dataset_fingerprint.json")
    plans = json.loads((prepared / "nnUNetPlans.json").read_text())
    if "3d_fullres" not in plans["configurations"]:
        raise ValueError("nnU-Net requires 3D volumes; this dataset produced no 3D plan.")
    (prepared / "dataset.json").write_text(json.dumps(dataset))
    status(root, 0.08, "Preprocessing training volumes with the nnU-Net plan.")
    runner.preprocess(c=("3d_fullres",), n_proc=(1,))
    # The dataloader consumes exactly these training cases. No random fold is
    # generated and no held-out case is passed to planning or optimization.
    cases = sorted(
        path.name.removesuffix(".nii.gz")
        for path in (root / "raw" / DATASET / "labelsTr").glob("*.nii.gz")
    )
    (prepared / "splits_final.json").write_text(json.dumps([{"train": cases, "val": []}]))
    trainer: Any = nnUNetTrainer(plans, "3d_fullres", 0, dataset, device=torch.device("cuda"))
    trainer.num_epochs = config.epochs
    trainer.initial_lr = config.learning_rate
    trainer.num_iterations_per_epoch = config.steps_per_epoch
    trainer.initialize()
    if request["mode"] == "continue":
        trainer.load_checkpoint(str(model / "fold_0/checkpoint_final.pth"))
    elif request["mode"] == "fine_tune":
        parent = torch.load(
            model / "fold_0/checkpoint_final.pth", map_location="cpu", weights_only=False
        )
        trainer.network.load_state_dict(parent["network_weights"], strict=True)
    first_epoch = trainer.current_epoch
    trainer.num_epochs = first_epoch + config.epochs
    trainer.lr_scheduler = PolyLRScheduler(
        trainer.optimizer, config.learning_rate, trainer.num_epochs
    )
    batches = training_batches(trainer)
    losses = []
    for epoch in range(first_epoch, trainer.num_epochs):
        trainer.on_train_epoch_start()
        outputs = []
        for step in range(config.steps_per_epoch):
            message = f"nnU-Net epoch {epoch - first_epoch + 1}/{config.epochs}."
            done = (epoch - first_epoch) * config.steps_per_epoch + step
            status(root, 0.15 + 0.8 * done / (config.epochs * config.steps_per_epoch), message)
            output = trainer.train_step(next(batches))
            if not np.isfinite(output["loss"]):
                raise ValueError("nnU-Net training produced a non-finite loss.")
            outputs.append(output)
        trainer.on_train_epoch_end(outputs)
        losses.append(float(np.mean([output["loss"] for output in outputs])))
        trainer.current_epoch += 1
    status(root, 0.96, "Saving nnU-Net weights, optimizer and preprocessing plan.")
    (model / "fold_0").mkdir(parents=True, exist_ok=True)
    # Upstream saves the next epoch, so account for our completed loop.
    trainer.current_epoch -= 1
    trainer.save_checkpoint(str(model / "fold_0/checkpoint_final.pth"))
    (model / "plans.json").write_text(json.dumps(plans))
    (model / "dataset.json").write_text(json.dumps(dataset))
    shutil.copyfile(prepared / "dataset_fingerprint.json", model / "dataset_fingerprint.json")
    (root / "training.json").write_text(
        json.dumps(
            {
                "completed_epochs": trainer.current_epoch + 1,
                "losses": losses,
                "configuration": "3d_fullres",
                "patch_size": trainer.configuration_manager.patch_size,
                "batch_size": trainer.batch_size,
            }
        )
    )


def predict(root: Path) -> None:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        device=torch.device("cuda"),
        use_mirroring=False,
        perform_everything_on_device=True,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(root / "model"), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    # Sequential preprocessing/export avoids extra process pools for one volume.
    predictor.predict_from_files_sequential(
        [[str(root / "image.nii.gz")]], [str(root / "mask")], save_probabilities=False
    )


def main() -> None:
    root = Path(sys.argv[1])
    request = json.loads((root / "request.json").read_text())
    if not torch.cuda.is_available():
        raise ValueError("nnU-Net v2 requires an available NVIDIA GPU.")
    torch.set_num_threads(4)
    if request["operation"] == "train":
        train(root, request)
    else:
        predict(root)
    status(root, 1, "nnU-Net finished.")


if __name__ == "__main__":
    main()
