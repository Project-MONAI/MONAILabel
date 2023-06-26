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

import os
import logging
import numpy as np
import torch
from tqdm import tqdm
import SimpleITK as sitk
from skimage import transform, io, segmentation
from lib.segment_anything import sam_model_registry
from lib.segment_anything.utils.transforms import ResizeLongestSide

logger = logging.getLogger(__name__)


class SAMEmbeddings:
    def __init__(self, checkpoint):
        self.device = 'cuda:0'
        self.image_size = 256
        self.model_type = 'vit_b'
        self.checkpoint = checkpoint
        # set up the SAM model
        self.sam_model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(self.device)
        if self.sam_model is None:
            logger.error('SAM pretrained model must be provided')

    def run(self, datastore, label_id):
        # Create npz folder
        embed_path = os.path.join(datastore._datastore_path, 'embeddings')
        os.makedirs(embed_path, exist_ok=True)
        if len(datastore.list_images()) == 0:
            logger.warning("Currently no images are available in datastore for SAM embeddings")
            return

        # List of all labeled images from the datastore
        all_lab_imgs = datastore.get_labeled_images()

        if len(all_lab_imgs) > 0:

            logger.info("Computing embeddings for SAM model...")
            for name in tqdm(all_lab_imgs):

                if os.path.exists(os.path.join(embed_path, name + f'_label_id_{label_id}.npz')):
                    logger.info(f'SAM embeddings already computed for volume {name} on label index {label_id}')
                    continue

                # Reading image
                image_sitk = sitk.ReadImage(datastore.get_image_uri(name))
                image_data = sitk.GetArrayFromImage(image_sitk)

                # Reading label/GT
                gt_sitk = sitk.ReadImage(datastore.get_label_uri(name, 'final'))
                gt_data = sitk.GetArrayFromImage(gt_sitk)

                # For training images
                imgs, gts, img_embeddings = self._preprocess_img(self, name, image_data, gt_data, label_id, self.sam_model, self.device)
                # Save npz
                self._save_npz(self, imgs, gts, img_embeddings, embed_path, name, label_id)

            logger.info(f"SAM embeddings complete!")
        else:
            logger.info(f"There is not labeled images - No SAM embeddings to compute!")

    @staticmethod
    def _preprocess_img(self, name, image_data, gt_data, label_id, sam_model, device):
        # Taking only the mask for label id
        gt_data[gt_data > label_id] = 0
        gt_data[gt_data < label_id] = 0
        # Binarizing mask
        gt_data[gt_data > 0] = 1
        if np.sum(gt_data) > 1000:
            imgs = []
            gts = []
            img_embeddings = []
            assert np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2, 'ground truth should be binary'
            # nii preprocess start
            lower_bound = -500
            upper_bound = 1000
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (image_data_pre - np.min(image_data_pre)) / (
                        np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
            image_data_pre[image_data == 0] = 0
            image_data_pre = np.uint8(image_data_pre)

            z_index, _, _ = np.where(gt_data > 0)
            z_min, z_max = np.min(z_index), np.max(z_index)

            for i in range(z_min, z_max):
                gt_slice_i = gt_data[i, :, :]
                gt_slice_i = transform.resize(gt_slice_i, (self.image_size, self.image_size), order=0, preserve_range=True,
                                              mode='constant', anti_aliasing=True)
                if np.sum(gt_slice_i) > 100:
                    # resize img_slice_i to 256x256
                    img_slice_i = transform.resize(image_data_pre[i, :, :], (self.image_size, self.image_size), order=3,
                                                   preserve_range=True, mode='constant', anti_aliasing=True)
                    # convert to three channels
                    img_slice_i = np.uint8(np.repeat(img_slice_i[:, :, None], 3, axis=-1))
                    assert len(img_slice_i.shape) == 3 and img_slice_i.shape[2] == 3, 'image should be 3 channels'
                    assert img_slice_i.shape[0] == gt_slice_i.shape[0] and img_slice_i.shape[1] == gt_slice_i.shape[
                        1], 'image and ground truth should have the same size'
                    imgs.append(img_slice_i)
                    assert np.sum(gt_slice_i) > 100, 'ground truth should have more than 100 pixels'
                    gts.append(gt_slice_i)
                    if sam_model is not None:
                        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                        resize_img = sam_transform.apply_image(img_slice_i)
                        # resized_shapes.append(resize_img.shape[:2])
                        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                        # model input: (1, 3, 1024, 1024)
                        input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
                        assert input_image.shape == (1, 3, sam_model.image_encoder.img_size,
                                                     sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                        with torch.no_grad():
                            embedding = sam_model.image_encoder(input_image)
                            img_embeddings.append(embedding.cpu().numpy()[0])

            return imgs, gts, img_embeddings

        else:
            logger.warning(f'Label id {self.label_id} in image {name} is too small for training')
            return [], [], []

    @staticmethod
    def _save_npz(self, imgs, gts, img_embeddings, embed_path, name, label_id):
        npz_path = os.path.join(embed_path, name + f'_label_id_{label_id}.npz')
        # save to npz file
        if len(imgs) > 1:
            imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
            gts = np.stack(gts, axis=0)  # (n, 256, 256)
            img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
            np.savez_compressed(npz_path, imgs=imgs, gts=gts, img_embeddings=img_embeddings)
            # save an example image for sanity check
            idx = np.random.randint(0, imgs.shape[0])
            img_idx = imgs[idx, :, :, :]
            gt_idx = gts[idx, :, :]
            bd = segmentation.find_boundaries(gt_idx, mode='inner')
            img_idx[bd, :] = [255, 0, 0]
            io.imsave(os.path.join(embed_path, name + f'_label_id_{label_id}.png'), img_idx, check_contrast=False)
