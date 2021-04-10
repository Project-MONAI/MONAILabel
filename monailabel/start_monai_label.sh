#!/bin/bash

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
app_dir=$(python3 ${DIR}/main.py --dryrun "$@" | grep "Using APP Directory" | cut -d'=' -f2)

if [ "${app_dir}" == "" ]; then
  return
fi

if [ -f ${app_dir}/requirements.txt ]; then
  if [ ! -d ${app_dir}/.venv ]; then
    python3 -m venv ${app_dir}/.venv
    source ${app_dir}/.venv/bin/activate

    python3 -m pip install --upgrade pip
    python3 -m pip install -r ${DIR}/../requirements.txt
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

export PYTHONPATH=${PYTHONPATH}:${app_dir}:${app_dir}/lib
echo "Using PYTHONPATH:: ${PYTHONPATH}"
python3 ${DIR}/main.py "$@"
deactivate

# TEST::
# ./start_monai_label.sh -a ../apps/my_app
