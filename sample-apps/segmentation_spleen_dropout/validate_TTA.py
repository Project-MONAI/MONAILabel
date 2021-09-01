import glob
import os
from time import sleep

import requests

'''

Workflow to compare TTA against random

1. Start training with 2 images
2. Perform TTA
3. Fetch image and retrain model
4. Perform TTA
5. Fetch image and retrain model

Questions:
- How to split val and train images
  In train class! Use the method "partition_datalist" to do the partition


- Why it is not working the epochs specification? I put 100 and it shows 50.
  In init_trainers method using config argument. BUT WHAT IS THE DIFFERENCE BETWEEN THAT AND THE EPOCHS IN REQUEST?

- How to specify I don't want to use pretrained model to start training?
  In init_trainers method. "load_from_mmar" is the method where the network is being specified


- A mix of object instantiation and API calls is not possible because
for API calls we'll need an IP to make the calls and
object instantiation doesn't have the option to stop the training, it executes line by line

- The disadvantage of using requests is that I need to first start the App via bash. Not everything is via PyCharm

THINGS TO IMPROVE:

- HAVE BETTER TRANSFORMS. IN TRAINING WE'RE USING CROPPING BUT NOT FOR INFERENCE. 
  FOR TRAINING WE'RE USING 96 PX AND FOR VALIDATION 160 PX. 
  
  DONE!!
  
- FOR SOME REASON THE THE MMARs MODEL DOESN'T WORK BETTER THAN 0.5 IN VALIDATION :/
  It seems the error comes from the transforms used in validation. The cropping doesn't allow to predict the whole mask??

'''

# # Start MONAI Label APP
# os.system("export PATH=$PATH:/home/adp20local/Documents/MONAILabel/monailabel/")
# os.system("monailabel start_server -a ./sample-apps/segmentation_spleen_tta/ -s /home/adp20local/Documents/Datasets/monailabel_datasets/spleen/train_small/")


# Getting server info
url_server = "http://127.0.0.1:8000/"
r = requests.get(url_server + "info/")
print(r.text)


new_labels = glob.glob("/home/adp20local/Documents/Datasets/monailabel_datasets/spleen/labels2train/*.nii.gz")

for idx in range(len(new_labels)):

    # Start training
    r = requests.post(url_server + "train/")
    print(r.text)

    # Waiting for some epochs
    for i in range(10):
        sleep(150)
        print("Training - Label number: " + str(idx) + " of " + str(len(new_labels)+1) + " -- Second: " + str(i + 1))

    # Stopping the training
    print('Stop training ---')
    r = requests.delete(url_server + "train/")
    print(r.text)

    sleep(5)

    # Start scoring
    # First experiment will wait until scoring ends. So we simulate like ideal scenario where scoring is immediate
    print('Start scoring ---')
    r = requests.post(url_server + "scoring/TTA")
    print(r.text)

    # Checking scoring status ---
    sleep(10)
    r = requests.get(url_server + "scoring/")

    while r.json()['status'] == 'RUNNING':
        sleep(180)
        print('Running scoring --- Checking every 180 seconds')
        r = requests.get(url_server + "scoring/")

    sleep(3)

    # Fetching an image
    print('Fetching image ---')
    r = requests.post(url_server + "activelearning/TTA")
    print(r.text)

    sleep(3)

    if r:
        # Copying the fetched image
        label_name = r.json()['id']
        print('Image fetched: ' + label_name)
        os.system(
            "cp /home/adp20local/Documents/Datasets/monailabel_datasets/spleen/labels2train/"
            + label_name
            + " /home/adp20local/Documents/Datasets/monailabel_datasets/spleen/train_small/labels/label_final_"
            + label_name
        )

    sleep(3)


# Re-starting training
r = requests.post(url_server + "train/")
print(r.text)
