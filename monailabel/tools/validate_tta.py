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

import argparse
import glob
import os
from time import sleep

import requests

parser = argparse.ArgumentParser(description="Active Learning Setting Using TTA")

# Directories & URL
parser.add_argument("--url_server", default="http://127.0.0.1:8000/", type=str)
# Path where labels used to simulate clinicians submission
parser.add_argument(
    "--path_new_labels",
    default="datasets/spleen/labels2train/",
    type=str,
)
# Path used to start the App + labels/
parser.add_argument(
    "--path_labels_root",
    default="datasets/spleen/train_small/labels/final/",
    type=str,
)

# Active learning parameters
parser.add_argument("--active_learning_technique", default="TTA", type=str)
# Factor used to wait until training happens
parser.add_argument("--training_time_factor", default=10, type=int)
parser.add_argument("--num_imgs_fetched", default=2, type=int)
args = parser.parse_args()

# First step is to start the App using the Terminal

# Getting server info - OPTIONAL
r = requests.get(args.url_server + "info/")
print(r.text)

new_labels = glob.glob(args.path_new_labels + "*.nii.gz")

for idx in range(len(new_labels)):

    # Start training
    r = requests.post(args.url_server + "train/")
    print(r.text)

    # Waiting to train model for some epochs
    for i in range(args.training_time_factor):
        sleep(3)
        print("Training - Label number: " + str(idx) + " of " + str(len(new_labels) + 1) + " -- Second: " + str(i + 1))

    # Stop training
    print("Stop training ---")
    r = requests.delete(args.url_server + "train/")
    print(r.text)

    sleep(5)

    # Start scoring
    # First experiment will wait until scoring ends. So we simulate like ideal scenario where scoring is immediate
    print("Start scoring ---")
    r = requests.post(args.url_server + "scoring/" + args.active_learning_technique)
    print(r.text)

    # Check scoring status ---
    sleep(5)
    r = requests.get(args.url_server + "scoring/")
    while r.json()["status"] == "RUNNING":
        sleep(180)
        print("Running scoring --- Checking every 180 seconds")
        r = requests.get(args.url_server + "scoring/")

    # Fetch images
    for _ in range(args.num_imgs_fetched):

        # Fetch an image using active learning technique score
        print("Fetching image ---")
        r = requests.post(args.url_server + "activelearning/" + args.active_learning_technique)
        print(r.text)

        sleep(2)

        if r:
            # Copy fetched image
            label_name = r.json()["name"]
            print("Image fetched: " + label_name)
            os.system("cp " + args.path_new_labels + label_name + " " + args.path_labels_root + label_name)

        sleep(2)

# Re-start training
r = requests.post(args.url_server + "train/")
print(r.text)
