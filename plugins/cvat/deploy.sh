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

# Sample commands to deploy MONAILabel nuclio functions
# ./deploy.sh <function> <model> <functions_dir>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
FUNCTION=${1:-**}
MODEL=${2:-*}
FUNCTIONS_DIR=${3:-$SCRIPT_DIR}
MONAI_LABEL_SERVER="${MONAI_LABEL_SERVER:-http://`hostname -I | awk '{print $1}'`:8000}"

nuctl create project cvat

shopt -s globstar

for func_config in "$FUNCTIONS_DIR"/$FUNCTION/${MODEL}.yaml
do
  func_root="$FUNCTIONS_DIR"
  echo "Using MONAI Label Server: $MONAI_LABEL_SERVER"
  cp $func_config ${func_config}.bak
  sed -i "s|http://monailabel.com|$MONAI_LABEL_SERVER|g" $func_config
  echo "Deploying $func_config..."
  nuctl deploy --project-name cvat --path "$func_root" --file "$func_config" --platform local
  mv ${func_config}.bak $func_config
done

nuctl get function
