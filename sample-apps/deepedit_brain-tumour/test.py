from monailabel.interfaces.test import test_main

if __name__ == "__main__":
    test_main()

"""
# Example commands to run inference or train locally

# Inference
python test.py -a . -s MSD_Task01_BrainTumour_single/imagesTr infer -m segmentation -i BRATS_001.nii.gz -o label_final_BRATS_001.nii.gz

# Train
python test.py -a . -s MSD_Task01_BrainTumour_single/imagesTr train -n model_01

"""
