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
    default="/home/adp20local/Documents/Datasets/monailabel_datasets/spleen/labels2train/",
    type=str,
)
# Path used to start the App + labels/
parser.add_argument(
    "--path_labels_root",
    default="/home/adp20local/Documents/Datasets/monailabel_datasets/spleen/train_small/labels/final/",
    type=str,
)

# Active learning parameters
parser.add_argument("--active_learning_technique", default="TTA", type=str)
# Factor used to wait until training happens
parser.add_argument("--training_time_factor", default=10, type=int)
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
        sleep(150)
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

    sleep(5)

    # Fetch an image using active learning technique score
    print("Fetching image ---")
    r = requests.post(args.url_server + "activelearning/" + args.active_learning_technique)
    print(r.text)

    sleep(5)

    if r:
        # Copy fetched image
        label_name = r.json()["id"]
        print("Image fetched: " + label_name)
        os.system("cp " + args.path_new_labels + label_name + " " + args.path_labels_root + "label_final_" + label_name)

    sleep(5)

# Re-start training
r = requests.post(args.url_server + "train/")
print(r.text)
