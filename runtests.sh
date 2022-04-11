#! /bin/bash

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

# script for running all tests
set -e

# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]]; then # stdout is a terminal
  separator=$'--------------------------------------------------------------------------------\n'
  blue="$(
    tput bold
    tput setaf 4
  )"
  green="$(
    tput bold
    tput setaf 2
  )"
  red="$(
    tput bold
    tput setaf 1
  )"
  noColor="$(tput sgr0)"
fi

# configuration values
doCoverage=false
doQuickTests=false
doNetTests=false
doDryRun=false
doUnitTests=false
doPytypeFormat=false
doCleanup=false

NUM_PARALLEL=1

PY_EXE=${MONAILABEL_PY_EXE:-$(which python3)}

function print_usage() {
  echo "runtests.sh [--codeformat] [--pytype]"
  echo "            [--unittests] [--net] [--dryrun] [-j number] [--clean] [--help] [--version]"
  echo ""
  echo "MONAILABEL testing utilities."
  echo ""
  echo "Examples:"
  echo "./runtests.sh --codeformat            # run static checks"
  echo "./runtests.sh --unittests             # run unit tests with code coverage"
  echo "./runtests.sh --net                   # run integration tests (monailabel PIP package should have been installed)"
  echo "./runtests.sh --clean                 # clean up temporary files and run \"${PY_EXE} setup.py develop --uninstall\"."
  echo ""
  echo "Python type check options:"
  echo "    --pytype          : perform \"pytype\" static type checks"
  echo "    -j, --jobs        : number of parallel jobs to run \"pytype\" (default $NUM_PARALLEL)"
  echo ""
  echo "MONAILABEL unit testing options:"
  echo "    -u, --unittests   : perform unit testing"
  echo "    --net             : perform integration testing"
  echo ""
  echo "Misc. options:"
  echo "    --dryrun          : display the commands to the screen without running"
  echo "    -f, --codeformat  : shorthand to run all code style and static analysis tests"
  echo "    -c, --clean       : clean temporary files from tests and exit"
  echo "    -h, --help        : show this help message and exit"
  echo "    -v, --version     : show MONAILABEL and system version information and exit"
  echo ""
  echo "${separator}For bug reports and feature requests, please file an issue at:"
  echo "    https://github.com/Project-MONAI/MONAILabel/issues/new/choose"
  echo ""
  echo "To choose an alternative python executable, set the environmental variable, \"MONAILABEL_PY_EXE\"."
  exit 1
}

function check_import() {
  echo "python: ${PY_EXE}"
  ${cmdPrefix}${PY_EXE} -c "import monailabel"
}

function print_version() {
  ${cmdPrefix}${PY_EXE} -c 'import monailabel; monailabel.print_config()'
}

function install_deps() {
  echo "Pip installing MONAILABEL development dependencies and compile MONAILABEL extensions..."
  ${cmdPrefix}${PY_EXE} -m pip install -r requirements-dev.txt
}

function clean_py() {
  TO_CLEAN="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
  echo "Removing temporary files in ${TO_CLEAN}"

  rm -rf sample-apps/*/logs
  rm -rf sample-apps/*/.venv
  rm -rf sample-apps/*/model/*
  rm -rf sample-apps/*/bin
  rm -rf tests/data/*
  rm -rf monailabel/endpoints/static/ohif
  rm -rf pytest.log
  rm -rf htmlcov
  rm -rf coverage.xml
  rm -rf junit
  rm -rf docs/build/
  rm -rf docs/source/apidocs/
  rm -rf test-output.xml
  rm -rf .coverage
  rm -rf .env

  find ${TO_CLEAN} -type f -name "*.py[co]" -delete
  find ${TO_CLEAN} -type f -name "*.so" -delete
  find ${TO_CLEAN} -type d -name "__pycache__" -delete
  find ${TO_CLEAN} -type d -name ".pytest_cache" -exec rm -r "{}" +
  find ${TO_CLEAN} -maxdepth 1 -type f -name ".coverage.*" -delete

  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".eggs" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "monailabel.egg-info" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "build" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "dist" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".pytype" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".coverage" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "__pycache__" -exec rm -r "{}" +
}

function torch_validate() {
  ${cmdPrefix}${PY_EXE} -c 'import torch; print(torch.__version__); print(torch.rand(5,3))'
}

function print_error_msg() {
  echo "${red}Error: $1.${noColor}"
  echo ""
}

function is_pip_installed() {
  return $(${PY_EXE} -c "import sys, pkgutil; sys.exit(0 if pkgutil.find_loader(sys.argv[1]) else 1)" $1)
}

if [ -z "$1" ]; then
  print_error_msg "Too few arguments to $0"
  print_usage
fi

# parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --net)
    doNetTests=true
    ;;
  --dryrun)
    doDryRun=true
    ;;
  -u | --u*) # allow --unittest | --unittests | --unittesting  etc.
    doUnitTests=true
    ;;
  -f | --codeformat)
    doPytypeFormat=true
    ;;
  --pytype)
    doPytypeFormat=true
    ;;
  -j | --jobs)
    NUM_PARALLEL=$2
    shift
    ;;
  -c | --clean)
    doCleanup=true
    ;;
  -h | --help)
    print_usage
    ;;
  -v | --version)
    print_version
    exit 1
    ;;
  --nou*) # allow --nounittest | --nounittests | --nounittesting  etc.
    print_error_msg "nounittest option is deprecated, no unit tests is the default setting"
    print_usage
    ;;
  *)
    print_error_msg "Incorrect commandline provided, invalid key: $key"
    print_usage
    ;;
  esac
  shift
done

# home directory
homedir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$homedir"

# python path
export PYTHONPATH="$homedir:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# by default do nothing
cmdPrefix=""

if [ $doDryRun = true ]; then
  echo "${separator}${blue}dryrun${noColor}"

  # commands are echoed instead of ran
  cmdPrefix="dryrun "
  function dryrun() { echo "    " "$@"; }
else
  check_import
fi

if [ $doCleanup = true ]; then
  echo "${separator}${blue}clean${noColor}"

  clean_py

  echo "${green}done!${noColor}"
  exit
fi

# unconditionally report on the state of monailabel
print_version

if [ $doPytypeFormat = true ]; then
  set +e # disable exit on failure so that diagnostics can be given on failure
  echo "${separator}${blue}pytype${noColor}"

  # ensure that the necessary packages for code format testing are installed
  if ! is_pip_installed pytype; then
    install_deps
  fi
  ${cmdPrefix}${PY_EXE} -m pytype --version

  ${cmdPrefix}${PY_EXE} -m pytype -j ${NUM_PARALLEL} --python-version="$(${PY_EXE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"

  pytype_status=$?
  if [ ${pytype_status} -ne 0 ]; then
    echo "${red}failed!${noColor}"
    exit ${pytype_status}
  else
    echo "${green}passed!${noColor}"
  fi
  set -e # enable exit on failure
fi

# testing command to run
cmd="${PY_EXE}"


# unit tests
if [ $doUnitTests = true ]; then
  echo "${separator}${blue}unittests${noColor}"
  torch_validate

  ${cmdPrefix}${PY_EXE} tests/setup.py
  ${cmdPrefix}${cmd} -m pytest -x --forked --doctest-modules --junitxml=junit/test-results.xml --cov-report xml --cov-report html --cov-report term --cov monailabel tests/unit
fi

function check_server_running() {
  local code=$(curl --write-out "%{http_code}\n" -s "http://127.0.0.1:${MONAILABEL_SERVER_PORT:-8000}/" --output /dev/null)
  echo ${code}
}

# network training/inference/eval integration tests
if [ $doNetTests = true ]; then
  echo "${separator}${blue}integration${noColor}"
  torch_validate

  ${cmdPrefix}${PY_EXE} tests/setup.py
  echo "Starting MONAILabel server..."
  rm -rf tests/data/apps
  monailabel apps -n radiology -o tests/data/apps -d
  monailabel start_server -a tests/data/apps/radiology -c models all -s tests/data/dataset/local/spleen -p ${MONAILABEL_SERVER_PORT:-8000} &

  wait_time=0
  server_is_up=0
  start_time_out=180

  while [[ $wait_time -le ${start_time_out} ]]; do
    if [ "$(check_server_running)" == "200" ]; then
      server_is_up=1
      break
    fi
    sleep 5
    wait_time=$((wait_time + 5))
    echo "Waiting for MONAILabel to be up and running..."
  done
  echo ""

  if [ "$server_is_up" == "1" ]; then
    echo "MONAILabel server is up and running."
  else
    echo "Failed to start MONAILabel server. Exiting..."
    exit 1
  fi

  ${cmdPrefix}${cmd} -m pytest -v tests/integration --no-summary -x
  echo "Finished All Integration Tests;  Stop/Kill MONAILabel Server..."
  kill -9 $(ps -ef | grep monailabel | grep start_server | grep -v grep | awk '{print $2}')
fi
