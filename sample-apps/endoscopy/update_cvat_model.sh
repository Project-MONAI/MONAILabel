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

APP_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Hash table for bundle names
declare -A BUNDLENAMES=( ["inbody"]="endoscopic_inbody_classification" ["tooltracking"]="endoscopic_tool_segmentation")

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

# Fetch latest model
if [ $FUNC_NAME == "deepedit" ];then
    MODEL_PATH="$APP_ROOT/model/$FUNC_NAME.pt"
    # Replace prior pretrained model with lastest model as current pre-trained model
    MODEL_CONTAINER="/opt/conda/monailabel/sample-apps/endoscopy/model/pretrained_$FUNC_NAME.pt" # default model path at function container
else
    # if bundle is used, get bundle name and fetch the model
    BUNDLE_NAME=${BUNDLENAMES[$FUNC_NAME]}
    MODEL_PATH="$APP_ROOT/model/$BUNDLE_NAME/models/model.pt"
    # Update to bundle nuclio container
    MODEL_CONTAINER="/opt/conda/monailabel/sample-apps/endoscopy/model/$BUNDLE_NAME/models/model.pt" # default model path at function container
fi

# Check if latest  model checkpoint is done and saved.
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "Latest model checkpoint not provided or published, exiting..."
else
    $(docker cp "$MODEL_PATH" "$FUNC_CONTAINER:$MODEL_CONTAINER")
    echo "Fetched and Published latest model: $FUNC_NAME to the nuclio function container."
fi
