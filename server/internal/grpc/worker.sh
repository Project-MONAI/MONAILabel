#!/bin/bash

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

app=$1
app_dir=$2
port=$3

if [ -f ${app_dir}/requirements.txt ]; then
  if [ ! -d ${app_dir}/.venv ]; then
    python3 -m venv ${app_dir}/.venv
    source ${app_dir}/.venv/bin/activate

    python3 -m pip install --upgrade pip
    python3 -m pip install -r ${DIR}/../../../requirements.txt
  else
    python3 -m venv ${app_dir}/.venv
    source ${app_dir}/.venv/bin/activate
  fi

  # Always update...
  python3 -m pip install --upgrade pip
  python3 -m pip install -r ${app_dir}/requirements.txt

  if [ $? -ne 0 ]; then
    echo "Failed to initialize APP venv"
    deactivate
    exit 1
  fi
fi

python3 ${DIR}/worker.py --name ${app} --path ${app_dir} --port ${port}
deactivate

# TEST::
# ./worker.sh segmentation_spleen ../../workspace/apps/segmentation_spleen 50051 &
# python request.py
