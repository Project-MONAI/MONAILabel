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
import argparse
import csv
import glob
import os

import monai
import torch
from monai.data import DataLoader
from monai.transforms import Compose, LoadImaged, NormalizeIntensityd, Resized


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument(
        "-i",
        "--input",
        default="/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeuroAtlas-Labels/DrTures/test-raw-dicom/",
    )
    args = parser.parse_args()

    images = glob.glob(os.path.join(args.input, "*/*/*"))

    # images = images[:10]

    num_labels = 4

    test_files = [{"img": img} for img in images]

    # define transforms for image
    test_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            NormalizeIntensityd(keys="img"),
            Resized(keys=["img"], spatial_size=(128, 128, 128)),
        ]
    )

    # create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_labels).to(device)

    # load params
    model.load_state_dict(torch.load("./runs_dict_DrTure/net_checkpoint_2200.pt")["net"])
    model.eval()

    # create a validation data loader
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    labels = ["FLAIR", "T1", "T1C", "T2"]
    with torch.no_grad():
        writer = csv.writer(open("preds.csv", "w"))
        dataset_iterator = test_loader._get_iterator()
        for x in range(len(images)):
            try:
                test_img = next(dataset_iterator)
                test_name = test_img["img_meta_dict"]["filename_or_obj"][0].split("/")[-3:]
                test_img = test_img["img"].to(device)
                output = model(test_img).argmax(dim=1)
                writer.writerow([test_name[0] + "/" + test_name[1] + "/" + test_name[2], labels[int(output.cpu())]])
                print(
                    f"Prediction for test_name {test_name[0]}/{test_name[1]}/{test_name[2]}: {labels[int(output.cpu())]}"
                )
            except:
                writer.writerow([test_name[0] + "/" + test_name[1] + "/" + test_name[2], "ERROR LOADING"])
                pass


if __name__ == "__main__":
    main()
