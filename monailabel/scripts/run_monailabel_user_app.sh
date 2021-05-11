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
  source .venv/bin/activate || (python3 -m venv .venv && source .venv/bin/activate)

  # always ensure the user packages are up to date
  python -m pip install --upgrade pip >/dev/null

  tmpfile=$(mktemp /tmp/requirements_XXXXXX.txt)
  cat "$(dirname "$(readlink -f "$0")")/user-requirements.txt" <(echo) requirements.txt >>${tmpfile}

  echo "++++ Installing PIP requirements"
  python -m pip install -r ${tmpfile} >/dev/null
  if [[ $? -ne 0 ]]; then
    echo "Failed to initialize APP venv"
    rm ${tmpfile}
    deactivate
    exit 1
  fi
  rm ${tmpfile}
fi

export PYTHONPATH=${PYTHONPATH}:${DIR}:${app_dir}
echo "Using PYTHONPATH:: ${PYTHONPATH}"
if [[ -z "$output_dir" ]]; then
  python -m monailabel.utils.others.app_utils -a "$app_dir" -s "$study_dir" "$user_command" -r "$train_request"
else
  python -m monailabel.utils.others.app_utils -a "$app_dir" -s "$study_dir" "$user_command" -r "$train_request" -o "$output_dir"
fi

if [[ -f requirements.txt ]]; then
  deactivate
fi

popd >/dev/null
