#!/bin/bash
echo "Installing requirements...."
apt-get install orthanc orthanc-dicomweb plastimatch -y

echo "You can convert NIFTI to DICOM using following utility"
echo plastimatch convert --patient-id patient1 --input image.nii.gz --output-dicom test

echo ""
echo ORTHANC default runs at: http://127.0.0.1:8042/app/explorer.html


# Upgrade to latest version
sudo service orthanc stop
sudo wget https://lsb.orthanc-server.com/orthanc/1.9.6/Orthanc --output-document /usr/sbin/Orthanc
sudo rm -f /usr/share/orthanc/plugins/*.so
sudo wget https://lsb.orthanc-server.com/orthanc/1.9.6/libServeFolders.so --output-document /usr/share/orthanc/plugins/libServeFolders.so
sudo wget https://lsb.orthanc-server.com/orthanc/1.9.6/libModalityWorklists.so --output-document /usr/share/orthanc/plugins/libModalityWorklists.so
sudo wget https://lsb.orthanc-server.com/plugin-dicom-web/1.6/libOrthancDicomWeb.so --output-document /usr/share/orthanc/plugins/libOrthancDicomWeb.so
sudo service orthanc restart
