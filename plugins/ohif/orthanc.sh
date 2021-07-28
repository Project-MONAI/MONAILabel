#!/bin/bash
echo "Installing requirements...."
apt-get install orthanc orthanc-dicomweb plastimatch -y

echo "You can convert NIFTI to DICOM using following utility"
echo plastimatch convert --patient-id patient1 --input image.nii.gz --output-dicom test

echo ""
echo ORTHANC default runs at: http://127.0.0.1:8042/app/explorer.html
