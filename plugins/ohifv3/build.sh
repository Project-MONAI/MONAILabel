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
git checkout feat/monai-label


sed -i "s|routerBasename: '/'|routerBasename: '/ohif/'|g" ./platform/app/public/config/default.js
sed -i "s|name: 'aws'|name: 'Orthanc'|g" ./platform/app/public/config/default.js
sed -i "s|wadoUriRoot: 'https://d33do7qe4w26qo.cloudfront.net/dicomweb'|wadoUriRoot: 'http://localhost/dicom-web'|g" ./platform/app/public/config/default.js
sed -i "s|wadoRoot: 'https://d33do7qe4w26qo.cloudfront.net/dicomweb'|wadoRoot: 'http://localhost/dicom-web'|g" ./platform/app/public/config/default.js
sed -i "s|qidoRoot: 'https://d33do7qe4w26qo.cloudfront.net/dicomweb'|qidoRoot: 'http://localhost/dicom-web'|g" ./platform/app/public/config/default.js

sed -i "s|PUBLIC_URL=/|PUBLIC_URL=/ohif/|g" ./platform/app/.env


yarn install

# Link the mode and extension HERE
echo "Linking extension and mode at: $(pwd)"
yarn run cli link-extension ../extension-monai-label
yarn run cli link-mode ../mode-monai-label

cd ../extension-monai-label

echo "Running install again at: $(pwd)"

yarn install

echo "Moving nrrd-js and itk node modules to Viewersnode_modules/"

cp -r ./node_modules/nrrd-js ../Viewers/node_modules/

cp -r ./node_modules/itk ../Viewers/node_modules/

echo "Moving to Viewers folder to build OHIF"

cd ../Viewers

echo "Viewers folder before building OHIF $(pwd)"

QUICK_BUILD=true yarn run build

rm -rf ${install_dir}
mv ./platform/app/dist/ ${install_dir}
echo "Copied OHIF to ${install_dir}"

rm -rf ../Viewers

cd ${curr_dir}
