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

mkdir -p logs

rm -rf www
mkdir -p www/html

cp -r ../../monailabel/endpoints/static/ohif www/html/ohif
cp -f config/monai_label.js www/html/ohif/app-config.js

nginx -p `pwd` -c config/nginx.conf -e logs/error.log
