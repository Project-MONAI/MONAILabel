#!/bin/bash
# Sample commands to deploy MONAILabel nuclio functions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR/pathology}

#nuctl create project cvat

shopt -s globstar
func_config="$FUNCTIONS_DIR/segmentation_nuclei.yaml"

func_root=$(dirname "$func_config")
echo "Deploying $func_config..."

echo nuctl deploy --project-name cvat --path "$func_root" --file "$func_config" --platform local
nuctl get function
