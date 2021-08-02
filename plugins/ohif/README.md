## MONAILabel Plugin for OHIF Viewer

![](Screenshots/1.png)

## Installing Orthanc (DICOMWeb)

### Ubuntu 20.x
```shell
# Install orthanc and dicomweb plugin
apt-get install orthanc orthanc-dicomweb -y

# Install plastimatch
apt-get install plastimatch -y
```

### Windows
 - Download and Install Orthanc from https://www.orthanc-server.com/download.php

## Converting NIFTI to DICOM
```shell
plastimatch convert --patient-id patient1 --input image.nii.gz --output-dicom test 
```