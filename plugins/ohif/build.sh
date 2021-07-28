#!/bin/bash
echo "Installing requirements...."
apt install npm -y
npm install --global yarn

curr_dir="$(pwd)"
my_dir="$(dirname "$(readlink -f "$0")")"

install_dir=${1:-$curr_dir/../../monailabel/static/ohif}

echo "Current Dir: ${curr_dir}"
echo "My Dir: ${my_dir}"
echo "Installing OHIF at: ${install_dir}"

cd ${my_dir}

if [ ! -d Viewers ]; then
  git clone https://github.com/OHIF/Viewers.git
fi

cd Viewers

# Viewers/platform/viewer/public/config/default.js
git checkout -- ./platform/viewer/public/config/default.js
sed -i "s|routerBasename: '/'|routerBasename: '/ohif/'|g" ./platform/viewer/public/config/default.js
sed -i "s|name: 'DCM4CHEE'|name: 'Orthanc'|g" ./platform/viewer/public/config/default.js
sed -i "s|https://server.dcmjs.org/dcm4chee-arc/aets/DCM4CHEE/wado|/proxy/dicom/wado|g" ./platform/viewer/public/config/default.js
sed -i "s|https://server.dcmjs.org/dcm4chee-arc/aets/DCM4CHEE/rs|/proxy/dicom/dicom-web|g" ./platform/viewer/public/config/default.js

# Viewers/platform/viewer/.env
git checkout -- ./platform/viewer/.env
sed -i "s|PUBLIC_URL=/|PUBLIC_URL=/ohif/|g" ./platform/viewer/.env

# monailabel plugin
cd extensions
rm monai-label
ln -s ../../monai-label monai-label
cd ..

git checkout -- ./platform/viewer/src/index.js
sed -i "s|let config = {};|import OHIFMONAILabelExtension from '@ohif/extension-monai-label';\nlet config = {};|g" ./platform/viewer/src/index.js
sed -i "s|defaultExtensions: \[|defaultExtensions: \[OHIFMONAILabelExtension,|g" ./platform/viewer/src/index.js

yarn config set workspaces-experimental true
yarn install
rm -rf ./Viewers/platform/viewer/dist
QUICK_BUILD=true yarn run build

# Reset if you want to run directly from yarn run dev:orthanc (without monailabel server)
git checkout -- platform/viewer/.env
git checkout -- platform/viewer/public/config/default.js
git checkout -- yarn.lock
cd ..

rm -rf ${install_dir}
mv ./Viewers/platform/viewer/dist ${install_dir}
echo "Copied OHIF to ${install_dir}"
cd ${curr_dir}
