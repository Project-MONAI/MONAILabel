import glob
import os
import shutil

data_dir = "/home/andres/Documents/workspace/disk-workspace/DatasetsMore/kidneyData/kits19-master/data/"
output_folder = "/home/andres/Documents/workspace/disk-workspace/DatasetsMore/kidneyData/kits19-master/monailabel/"

all_files = glob.glob(os.path.join(data_dir, "*"))


for idx, img_path in enumerate(all_files):
    fname = img_path.split("/")[-1]
    print(f"Processing image: {idx}/{len(all_files)}")
    img_gt = glob.glob(os.path.join(data_dir, fname, '*.nii.gz'))
    for l in img_gt:
        f = l.split("/")[-1]
        if 'segmentation' in f:
            shutil.copy(l, output_folder + 'labels/final/' + fname + ".nii.gz")
        elif 'imaging' in f:
            shutil.copy(l, output_folder + fname + ".nii.gz")

