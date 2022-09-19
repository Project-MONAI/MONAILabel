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

# Sample commands to publish trained latest model to nuclio functions
# bash /bin/update_cvat_model.sh <FUNCTION_NAME> <MODEL_PATH>

FUNC_NAME=$1
MODEL_PATH=$2

if [ $# -eq 0 ];then
    echo "Usage: No function name provided, exiting..."
    exit 1
fi

# get function container name by argument
FUNC_CONTAINER="nuclio-nuclio-monailabel.endoscopy.$FUNC_NAME"

# check if function containers are running
if [ ! $(docker inspect -f '{{.State.Status}}' $FUNC_CONTAINER) == "running" ]; then
    echo "$FUNC_CONTAINER container is not running, can not publish to container..."
fi

# default published model name
MODEL_CONTAINER="/opt/conda/monailabel/sample-apps/endoscopy/model/pretrained_$FUNC_NAME.pt" # default model path at function container

# Check if latest  model checkpoint is done and saved, error if blank.
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "Latest model checkpoint not provided or published, exiting..."
else
    # Replace prior pretrained model with lastest model as current pre-trained model
    $(docker cp "$MODEL_PATH" "$FUNC_CONTAINER:$MODEL_CONTAINER")
    echo "Published latest mode: $MODEL_PATH into $FUNC_NAME nuclio function container."
fi

