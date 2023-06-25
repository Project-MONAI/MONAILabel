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

import logging
import numpy as np
from monai.transforms import LoadImage
from tqdm import tqdm

import os
from skimage import transform
import torch
from lib.segment_anything import SamPredictor, sam_model_registry
from lib.segment_anything.utils.transforms import ResizeLongestSide

logger = logging.getLogger(__name__)


class SAMEmbeddings:
    def __init__(self, checkpoint):
        self.device = 'cuda:0'
        self.image_size = 256
        self.label_id = 5
        self.model_type = 'vit_b'
        self.checkpoint = checkpoint
        # %% set up the model
        self.sam_model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(self.device)

    def run(self, datastore):
        logger.info("Computing embeddings for SAM model...")
        # Create npz folder
        embed_path = os.path.join(datastore._datastore_path, 'embeddings')
        os.makedirs(embed_path, exist_ok=True)
        if len(datastore.list_images()) == 0:
            logger.warning("Currently no images are available in datastore for SAM embeddings")
            return

        # List of all images from the datastore
        all_imgs = datastore.list_images()

        loader = LoadImage()
        for name in tqdm(all_imgs):
            npz_path = os.path.join(embed_path, name + '.npz')
            img, _ = loader(datastore.get_image_uri(name))
            imgs, _ = self._preprocess_img(self, img, self.sam_model, self.device)
            # Save npz
            self._save_npz(self, imgs, npz_path)

        logger.info(f"SAM embeddings complete!")

    @staticmethod
    def _preprocess_img(self, image_data, sam_model, device):
        imgs = []
        img_embeddings = []
        # nii preprocess start
        lower_bound = -500
        upper_bound = 1000
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre)) / (
                    np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
        image_data_pre[image_data == 0] = 0
        image_data_pre = np.uint8(image_data_pre)

        z_index, _, _ = np.where(image_data > 0)
        z_min, z_max = np.min(z_index), np.max(z_index)

        for i in range(z_min, z_max):
            gt_slice_i = image_data[i, :, :]
            gt_slice_i = transform.resize(gt_slice_i, (self.image_size, self.image_size), order=0, preserve_range=True,
                                          mode='constant', anti_aliasing=True)
            # resize img_slice_i to 256x256
            img_slice_i = transform.resize(image_data_pre[i, :, :], (self.image_size, self.image_size), order=3,
                                           preserve_range=True, mode='constant', anti_aliasing=True)
            # convert to three channels
            img_slice_i = np.uint8(np.repeat(img_slice_i[:, :, None], 3, axis=-1))
            assert len(img_slice_i.shape) == 3 and img_slice_i.shape[2] == 3, 'image should be 3 channels'
            assert img_slice_i.shape[0] == gt_slice_i.shape[0] and img_slice_i.shape[1] == gt_slice_i.shape[
                1], 'image and ground truth should have the same size'
            imgs.append(img_slice_i)
            if sam_model is not None:
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(img_slice_i)
                # resized_shapes.append(resize_img.shape[:2])
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                # model input: (1, 3, 1024, 1024)
                input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
                assert input_image.shape == (1, 3, sam_model.image_encoder.img_size,
                                             sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                # input_imgs.append(input_image.cpu().numpy()[0])
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])

        if sam_model is not None:
            return imgs, img_embeddings
        else:
            return imgs

    @staticmethod
    def _save_npz(self, imgs, npz_path):
        # save to npz file
        if len(imgs) > 1:
            imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
            np.savez_compressed(npz_path, imgs=imgs)
