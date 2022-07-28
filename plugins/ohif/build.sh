#!/bin/bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

curr_dir="$(pwd)"
my_dir="$(dirname "$(readlink -f "$0")")"

echo "Installing requirements..."
sh $my_dir/requirements.sh

install_dir=${1:-$my_dir/../../monailabel/endpoints/static/ohif}

echo "Current Dir: ${curr_dir}"
echo "My Dir: ${my_dir}"
echo "Installing OHIF at: ${install_dir}"

cd ${my_dir}
rm -rf Viewers
git clone https://github.com/OHIF/Viewers.git
cd Viewers
git checkout 460fdeb534cd94bff55892c8e3d7100ccf8957de

# Viewers/platform/viewer/public/config/default.js
#git checkout -- ./platform/viewer/public/config/default.js
sed -i "s|routerBasename: '/'|routerBasename: '/ohif/'|g" ./platform/viewer/public/config/default.js
sed -i "s|name: 'DCM4CHEE'|name: 'Orthanc'|g" ./platform/viewer/public/config/default.js
sed -i "s|wadoUriRoot: 'https://server.dcmjs.org/dcm4chee-arc/aets/DCM4CHEE/wado'|wadoUriRoot: '/proxy/dicom/wado'|g" ./platform/viewer/public/config/default.js
sed -i "s|wadoRoot: 'https://server.dcmjs.org/dcm4chee-arc/aets/DCM4CHEE/rs'|wadoRoot: '/proxy/dicom/wado'|g" ./platform/viewer/public/config/default.js
sed -i "s|qidoRoot: 'https://server.dcmjs.org/dcm4chee-arc/aets/DCM4CHEE/rs'|qidoRoot: '/proxy/dicom/qido'|g" ./platform/viewer/public/config/default.js

# Viewers/platform/viewer/.env
#git checkout -- ./platform/viewer/.env
sed -i "s|PUBLIC_URL=/|PUBLIC_URL=/ohif/|g" ./platform/viewer/.env

# monailabel plugin
cd extensions
rm monai-label
ln -s ../../monai-label monai-label
cd ..

#git checkout -- ./platform/viewer/src/index.js
sed -i "s|let config = {};|import OHIFMONAILabelExtension from '@ohif/extension-monai-label';\nlet config = {};|g" ./platform/viewer/src/index.js
sed -i "s|defaultExtensions: \[|defaultExtensions: \[OHIFMONAILabelExtension,|g" ./platform/viewer/src/index.js

yarn config set workspaces-experimental true
yarn install
rm -rf ./Viewers/platform/viewer/dist
QUICK_BUILD=true yarn run build

# Reset if you want to run directly from yarn run dev:orthanc (without monailabel server)
#git checkout -- platform/viewer/.env
#git checkout -- platform/viewer/public/config/default.js
#git checkout -- yarn.lock

cd ..

rm -rf ${install_dir}
mv ./Viewers/platform/viewer/dist ${install_dir}
echo "Copied OHIF to ${install_dir}"

rm -rf Viewers
#git restore Viewers

cd ${curr_dir}
