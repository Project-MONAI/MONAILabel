## MONAILabel Plugin for OHIF Viewer

![](Screenshots/1.png)

## Development setup

You can build the OHIF plugin for development as follows:

```shell
(cd plugins/ohif && ./build.sh)

# If you want to avoid building OHIF package for every code changes, 
# you can run OHIF Viewer directly in checked-out git submodule
cd plugins/ohif/Viewers

yarn run dev:orthanc
```

## Installing Orthanc (DICOMWeb)

### Ubuntu 20.x

```shell
# Install orthanc and dicomweb plugin
sudo apt-get install orthanc orthanc-dicomweb -y

# Install plastimatch
sudo apt-get install plastimatch -y
```

However, you have to **upgrade to latest version** by following steps
mentioned [here](https://book.orthanc-server.com/users/debian-packages.html#replacing-the-package-from-the-service-by-the-lsb-binaries)

```shell
sudo service orthanc stop
sudo wget https://lsb.orthanc-server.com/orthanc/1.9.7/Orthanc --output-document /usr/sbin/Orthanc
sudo rm -f /usr/share/orthanc/plugins/*.so

sudo wget https://lsb.orthanc-server.com/orthanc/1.9.7/libServeFolders.so --output-document /usr/share/orthanc/plugins/libServeFolders.so
sudo wget https://lsb.orthanc-server.com/orthanc/1.9.7/libModalityWorklists.so --output-document /usr/share/orthanc/plugins/libModalityWorklists.so
sudo wget https://lsb.orthanc-server.com/plugin-dicom-web/1.6/libOrthancDicomWeb.so --output-document /usr/share/orthanc/plugins/libOrthancDicomWeb.so

sudo service orthanc restart
```

### Windows/Others _(latest version)_

- Download and Install Orthanc from https://www.orthanc-server.com/download.php

## Converting NIFTI to DICOM

```shell
plastimatch convert --patient-id patient1 --input image.nii.gz --output-dicom test 
```

## Uploading DICOM to Orthanc

### Use Orthanc Browser

Use orthanc browser located at http://127.0.0.1:8042/app/explorer.html#upload to upload the files.

### Using STORE SCP/SCU

#### Enable AET

`sudo vim /etc/orthanc/orthanc.json`

```json5
// The list of the known DICOM modalities
"DicomModalities" : {
/**
 * Uncommenting the following line would enable Orthanc to
 * connect to an instance of the "storescp" open-source DICOM
 * store (shipped in the DCMTK distribution) started by the
 * command line "storescp 2000".
 **/
"sample": ["MONAILABEL", "127.0.0.1", 104]
```

`sudo service orthanc restart`

#### Upload Files

```shell
# If AET 'MONAILABEL' is enabled in Orthanc
python -m pynetdicom storescu 127.0.0.1 4242 test -aet MONAILABEL -r
```