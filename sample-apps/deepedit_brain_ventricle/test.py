from monailabel.interfaces.test import test_main

if __name__ == "__main__":
    test_main()

"""
# Example commands to run inference or train locally

# Inference
python test.py -a . -s TaskXX_BrainVentricle/imagesTr infer -m segmentation \
       -i brain_ventricle.nii.gz -o label_final_brain_ventricle.nii.gz

# Train
python test.py -a . -s TaskXX_BrainVentricle/imagesTr train -n model_01

"""
