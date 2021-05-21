from monailabel.interfaces.test import test_main

if __name__ == "__main__":
    test_main()

"""
# Example commands to run inference or train locally

# Inference
python test.py -a . -s MSD_Task02_Heart/imagesTr infer -m segmentation -i la_004.nii.gz -o label.nii.gz

# Train
python test.py -a . -s MSD_Task02_Heart/imagesTr train -n model_01

"""
