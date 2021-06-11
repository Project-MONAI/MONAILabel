from monailabel.interfaces.test import test_main

if __name__ == "__main__":
    test_main()

"""
# Example commands to run inference or train locally

# Inference
python test.py -a . -s Task0X_Lung/imagesTr infer -m segmentation -i coronacases_001.nii.gz -o label.nii.gz

# Train
python test.py -a . -s Task0X_Lung/imagesTr train -n model_01

"""
