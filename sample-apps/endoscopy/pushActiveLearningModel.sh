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

# Sample commands to publish active learning trained latest model to nuclio functions
# ./pushActiveLearningModel.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

FUNCTIONS_DIR=${2:-$SCRIPT_DIR}

# only support tooltracking for now, deid, 
# to add loop checks for all containers/functions and published models, then push to containers. 
FUNC_NAME="tooltracking"

FUNC_CONTAINER="nuclio-nuclio-monailabel.endoscopy.$FUNC_NAME" #default function container name

# check if function containers are running
if [ ! $(docker inspect -f '{{.State.Status}}' $FUNC_CONTAINER) == "running" ]; then
    echo "$FUNC_CONTAINER container is not running, can not publish to container..."
fi 

CKP_FILE="$FUNCTIONS_DIR"/model/$FUNC_NAME.pt # default published model name
MODEL_CONTAINER="/opt/conda/monailabel/sample-apps/endoscopy/model/pretrained_$FUNC_NAME.pt" # default model path at function container

# Check if latest active learning model checkpoint is done and saved.
if [ ! -f $CKP_FILE ]; then
    echo "Active learning model checkpoint not published, exiting..."
else
    # Replace prior pretrained model with lastest model as current pre-trained model
    $(docker cp "$CKP_FILE" "$FUNC_CONTAINER:$MODEL_CONTAINER")
    echo "Published latest mode: $CKP_FILE into $FUNC_NAME nuclio function container..."
fi