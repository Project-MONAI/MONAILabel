#!/bin/bash
# Sample commands to deploy MONAILabel nuclio functions
# ./deploy.sh <function> <model> <functions_dir>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
FUNCTION=${1:-**}
MODEL=${2:-*}
FUNCTIONS_DIR=${3:-$SCRIPT_DIR}

nuctl create project cvat

shopt -s globstar

for func_config in "$FUNCTIONS_DIR"/$FUNCTION/${MODEL}.yaml
do
  func_root="$FUNCTIONS_DIR"
  echo "Deploying $func_config..."
  nuctl deploy --project-name cvat --path "$func_root" --file "$func_config" --platform local
done

nuctl get function
