#!/usr/bin/env bash
set -e

app_dir=$1
study_dir=$2
user_command=$3
train_request=$4
output_dir=$5

if [[ "${app_dir}" == "" ]]; then
  exit 1
fi

pushd "$app_dir" >/dev/null

if [[ -f requirements.txt ]]; then
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv 
  fi

  # try to activate an existing environment, or re-create an try to activate
  # if unable to activate the existing environment
  source .venv/bin/activate || (python3 -m venv --system-site-packages .venv && source .venv/bin/activate)

  # always ensure the user packages are up to date
  python3 -m pip install --upgrade pip >/dev/null

  echo "+++++++++++++++++ Installing PIP requirements"
  python3 -m pip install -r requirements.txt
  if [[ $? -ne 0 ]]; then
    echo "Failed to initialize APP venv"
    exit 1
  fi
fi

popd >/dev/null

export PYTHONPATH=${PYTHONPATH}:${DIR}:${app_dir}
echo "Using PYTHONPATH:: ${PYTHONPATH}"
if [[ -z "$output_dir" ]]; then
  python3 -m monailabel.utils.others.app_utils -a "$app_dir" -s "$study_dir" "$user_command" -r "$train_request"
else
  python3 -m monailabel.utils.others.app_utils -a "$app_dir" -s "$study_dir" "$user_command" -r "$train_request" -o "$output_dir"
fi
