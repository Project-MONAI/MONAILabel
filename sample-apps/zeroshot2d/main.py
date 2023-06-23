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

import json
import logging
import os
from typing import Dict

import lib.configs
from lib.activelearning import Last
from lib.infers.deepgrow_pipeline import InferDeepgrowPipeline
from lib.infers.vertebra_pipeline import InferVertebraPipeline

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.scribbles.infer import GMMBasedGraphCut, HistogramBasedGraphCut
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random

# bundle
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import get_bundle_models, strtobool
from monailabel.utils.others.planner import HeuristicPlanner

# SAM processing
import SimpleITK as sitk
import numpy as np
from skimage import transform, segmentation
from segment_anything.utils.transforms import ResizeLongestSide
import torch
# note: this import is only for preprocessing.
# training and inferring use SAM class in MONAI
# TODO: optimize
from segment_anything import sam_model_registry 
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}

        # Load models from app model implementation, e.g., --conf models <segmentation_spleen>
        models = conf.get("models")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        # app models
        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, self.planner)
        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        # Load models from bundle config files, local or released in Model-Zoo, e.g., --conf bundles <spleen_ct_segmentation>
        self.bundles = get_bundle_models(app_dir, conf, conf_key="bundles") if conf.get("bundles") else None

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Zeroshot2D ({monailabel.__version__})",
            description="Zeroshot 2D models for segmentation",
            version=monailabel.__version__,
        )
    
    def SAM_data_preprocessing(self):

        # TODO: embeddings
        # TODO: move to a lib class 
        # TODO: determine config location: pass in arguements, interface/config.py or use app config (Config class)
        def preprocess_ct(gt_path, nii_path, gt_name, image_name, label_id, image_size, sam_model, device):
            gt_sitk = sitk.ReadImage(os.path.join(gt_path, gt_name))
            gt_data = sitk.GetArrayFromImage(gt_sitk)
            gt_data = np.uint8(gt_data==label_id)
            if np.sum(gt_data)>1000:
                imgs, gts, img_embeddings = [], [], []
                assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
                img_sitk = sitk.ReadImage(os.path.join(nii_path, image_name))
                image_data = sitk.GetArrayFromImage(img_sitk)
                # nii preprocess start
                lower_bound = -500
                upper_bound = 1000
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
                image_data_pre[image_data==0] = 0
                image_data_pre = np.uint8(image_data_pre)
                z_index, _, _ = np.where(gt_data>0)
                z_min, z_max = np.min(z_index), np.max(z_index)
                for i in range(z_min, z_max):
                    gt_slice_i = gt_data[i,:,:]
                    gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
                    if np.sum(gt_slice_i)>100:
                        # resize img_slice_i to 256x256
                        img_slice_i = transform.resize(image_data_pre[i,:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
                        # convert to three channels
                        img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
                        assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
                        assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
                        imgs.append(img_slice_i)
                        assert np.sum(gt_slice_i)>100, 'ground truth should have more than 100 pixels'
                        gts.append(gt_slice_i)
                        if sam_model is not None:
                            sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                            resize_img = sam_transform.apply_image(img_slice_i)
                            # resized_shapes.append(resize_img.shape[:2])
                            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                            # model input: (1, 3, 1024, 1024)
                            input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
                            assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                            # input_imgs.append(input_image.cpu().numpy()[0])
                            with torch.no_grad():
                                embedding = sam_model.image_encoder(input_image)
                                img_embeddings.append(embedding.cpu().numpy()[0])
            if sam_model is not None:
                return imgs, gts, img_embeddings
            else:
                return imgs, gts
            
        # prepare the save path
        # TODO: set path following monai label app convent
        save_path_tr = os.path.join( ... ) # train
        save_path_ts = os.path.join( ... ) # test
        os.makedirs(save_path_tr, exist_ok=True)
        os.makedirs(save_path_ts, exist_ok=True)

        # set up the model
        # TODO: set model type following monai label app convent
        sam_model = sam_model_registry[ ... ](checkpoint= ... ).to( ... )

        # TODO: follow monai label division. (if any? I didn't see)
        # split names into training and testing
        prefix = args.modality + '_' + args.anatomy
        names = sorted(os.listdir(args.gt_path))
        names = [name for name in names if not os.path.exists(join(args.npz_path, prefix + '_' + name.split('.nii.gz')[0]+'.npz'))]
        names = [name for name in names if os.path.exists(join(args.nii_path, name.split('.nii.gz')[0] + args.img_name_suffix))]
        np.random.seed( ... )
        np.random.shuffle(names)
        train_names = sorted(names[:int(len(names)*0.8)])
        test_names = sorted(names[int(len(names)*0.8):])

        for name in tqdm(train_names):
            image_name = name.split('.nii.gz')[0] + args.img_name_suffix
            gt_name = name 
            imgs, gts, img_embeddings = preprocess_ct(args.gt_path, args.nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model, args.device)
            # save to npz file
            # stack the list to array
            if len(imgs)>1:
                imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
                gts = np.stack(gts, axis=0) # (n, 256, 256)
                img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
                np.savez_compressed(os.path.join(save_path_tr, prefix + '_' + gt_name.split('.nii.gz')[0]+'.npz'), imgs=imgs, gts=gts, img_embeddings=img_embeddings)
                # save an example image for sanity check
                idx = np.random.randint(0, imgs.shape[0])
                img_idx = imgs[idx,:,:,:]
                gt_idx = gts[idx,:,:]
                bd = segmentation.find_boundaries(gt_idx, mode='inner')
                img_idx[bd, :] = [255, 0, 0]
                io.imsave(save_path_tr + '.png', img_idx, check_contrast=False)

        # save testing data
        for name in tqdm(test_names):
            image_name = name.split('.nii.gz')[0] + args.img_name_suffix
            gt_name = name 
            imgs, gts = preprocess_ct(args.gt_path, args.nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model=None, device=args.device)
            # save to npz file
            if len(imgs)>1:
                imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
                gts = np.stack(gts, axis=0) # (n, 256, 256)
                img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
                np.savez_compressed(os.path.join(save_path_ts, prefix + '_' + gt_name.split('.nii.gz')[0]+'.npz'), imgs=imgs, gts=gts)
                # save an example image for sanity check
                idx = np.random.randint(0, imgs.shape[0])
                img_idx = imgs[idx,:,:,:]
                gt_idx = gts[idx,:,:]
                bd = segmentation.find_boundaries(gt_idx, mode='inner')
                img_idx[bd, :] = [255, 0, 0]
                io.imsave(save_path_ts + '.png', img_idx, check_contrast=False)

        return

    def init_datastore(self) -> Datastore:
        
        datastore = super().init_datastore()
        datastore.extensions=("*.nii.gz", "*.nii", "*.npz")
        # TODO: do the ResizeLongestSide
        self.SAM_data_preprocessing()
        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}

        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v

        #################################################
        # Bundle Models
        #################################################
        if self.bundles:
            for n, b in self.bundles.items():
                i = BundleInferTask(b, self.conf)
                logger.info(f"+++ Adding Bundle Inferer:: {n} => {i}")
                infers[n] = i

        #################################################
        # Scribbles
        #################################################
        infers.update(
            {
                "Histogram+GraphCut": HistogramBasedGraphCut(
                    intensity_range=(-300, 200, 0.0, 1.0, True),
                    pix_dim=(2.5, 2.5, 5.0),
                    lamda=1.0,
                    sigma=0.1,
                    num_bins=64,
                    labels=task_config.labels,
                ),
                "GMM+GraphCut": GMMBasedGraphCut(
                    intensity_range=(-300, 200, 0.0, 1.0, True),
                    pix_dim=(2.5, 2.5, 5.0),
                    lamda=5.0,
                    sigma=0.5,
                    num_mixtures=20,
                    labels=task_config.labels,
                ),
            }
        )

        #################################################
        # Pipeline based on existing infers
        #################################################
        if infers.get("deepgrow_2d") and infers.get("deepgrow_3d"):
            infers["deepgrow_pipeline"] = InferDeepgrowPipeline(
                path=self.models["deepgrow_2d"].path,
                network=self.models["deepgrow_2d"].network,
                model_3d=infers["deepgrow_3d"],
                description="Combines Clara Deepgrow 2D and 3D models",
            )

        #################################################
        # Pipeline based on existing infers for vertebra segmentation
        # Stages:
        # 1/ localization spine
        # 2/ localization vertebra
        # 3/ segmentation vertebra
        #################################################
        if (
            infers.get("localization_spine")
            and infers.get("localization_vertebra")
            and infers.get("segmentation_vertebra")
        ):
            infers["vertebra_pipeline"] = InferVertebraPipeline(
                task_loc_spine=infers["localization_spine"],  # first stage
                task_loc_vertebra=infers["localization_vertebra"],  # second stage
                task_seg_vertebra=infers["segmentation_vertebra"],  # third stage
                description="Combines three stage for vertebra segmentation",
            )
        logger.info(infers)
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            t = task_config.trainer()
            if not t:
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t

        #################################################
        # Bundle Models
        #################################################
        if self.bundles:
            for n, b in self.bundles.items():
                t = BundleTrainTask(b, self.conf)
                if not t or not t.is_valid():
                    continue

                logger.info(f"+++ Adding Bundle Trainer:: {n} => {t}")
                trainers[n] = t

        return trainers

"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse
    import shutil
    from pathlib import Path

    from monailabel.utils.others.generic import device_list, file_ext

    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Dataset/Zeroshot2D"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    parser.add_argument("-m", "--model", default="segmentation_sam")
    parser.add_argument("-t", "--test", default="infer", choices=("train", "infer"))
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "models": args.model,
        "preload": "true",
    }

    app = MyApp(app_dir, studies, conf)

    # Infer
    if args.test == "infer":
        sample = app.next_sample(request={"strategy": "first"})
        image_id = sample["id"]
        image_path = sample["path"]

        # Run on all devices
        for device in device_list():
            res = app.infer(request={"model": args.model, "image": image_id, "device": device})
            # res = app.infer(
            #     request={"model": "vertebra_pipeline", "image": image_id, "device": device, "slicer": False}
            # )
            label = res["file"]
            label_json = res["params"]
            test_dir = os.path.join(args.studies, "test_labels")
            os.makedirs(test_dir, exist_ok=True)

            label_file = os.path.join(test_dir, image_id + file_ext(image_path))
            shutil.move(label, label_file)

            print(label_json)
            print(f"++++ Image File: {image_path}")
            print(f"++++ Label File: {label_file}")
            break
        return

    # Train
    app.train(
        request={
            "model": args.model,
            "max_epochs": 10,
            "dataset": "Dataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 1,
            "val_batch_size": 1,
            "multi_gpu": False,
            "val_split": 0.1,
        },
    )


if __name__ == "__main__":
    # export PYTHONPATH=~/Projects/MONAILabel:`pwd`
    # python main.py
    main()
