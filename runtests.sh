#! /bin/bash

# Copyright 2020 - 2021 MONAI Consortium
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
doZooTests=false
doUnitTests=false
doBlackFormat=false
doBlackFix=false
doIsortFormat=false
doIsortFix=false
doFlake8Format=false
doClangFormat=false
doPytypeFormat=false
doMypyFormat=false
doCleanup=false

NUM_PARALLEL=1
LINE_LENGTH=120

PY_EXE=${MONAILABEL_PY_EXE:-$(which python3)}

function print_usage() {
  echo "runtests.sh [--codeformat] [--autofix] [--isort] [--flake8] [--pytype] [--mypy]"
  echo "            [--unittests] [--coverage] [--quick] [--net] [--dryrun] [-j number] [--clean] [--help] [--version]"
  echo ""
  echo "MONAILABEL unit testing utilities."
  echo ""
  echo "Examples:"
  echo "./runtests.sh -f -u --net --coverage  # run style checks, full tests, print code coverage (${green}recommended for pull requests${noColor})."
  echo "./runtests.sh -f -u                   # run style checks and unit tests."
  echo "./runtests.sh -f                      # run coding style and static type checking."
  echo "./runtests.sh --quick --unittests     # run minimal unit tests, for quick verification during code developments."
  echo "./runtests.sh --autofix               # run automatic code formatting using \"isort\" and \"black\"."
  echo "./runtests.sh --clean                 # clean up temporary files and run \"${PY_EXE} setup.py develop --uninstall\"."
  echo ""
  echo "Code style check options:"
  echo "    --black           : perform \"black\" code format checks"
  echo "    --autofix         : format code using \"isort\" and \"black\""
  echo "    --isort           : perform \"isort\" import sort checks"
  echo "    --flake8          : perform \"flake8\" code format checks"
  echo ""
  echo "Python type check options:"
  echo "    --pytype          : perform \"pytype\" static type checks"
  echo "    --mypy            : perform \"mypy\" static type checks"
  echo "    -j, --jobs        : number of parallel jobs to run \"pytype\" (default $NUM_PARALLEL)"
  echo ""
  echo "MONAILABEL unit testing options:"
  echo "    -u, --unittests   : perform unit testing"
  echo "    --coverage        : report testing code coverage, to be used with \"--net\", \"--unittests\""
  echo "    -q, --quick       : skip long running unit tests and integration tests"
  echo "    --net             : perform integration testing"
  echo "    --list_tests      : list unit tests and exit"
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
  # remove coverage history
  # ${cmdPrefix}${PY_EXE} -m coverage erase

  # uninstall the development package
  # echo "Uninstalling MONAILABEL development files..."
  # ${cmdPrefix}${PY_EXE} setup.py develop --user --uninstall

  # remove temporary files (in the directory of this script)
  TO_CLEAN="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
  echo "Removing temporary files in ${TO_CLEAN}"

	rm -rf sample-apps/*/logs
	rm -rf sample-apps/*/.venv
	rm -rf sample-apps/*/model/*

  find ${TO_CLEAN}/monailabel -type f -name "*.py[co]" -delete
  find ${TO_CLEAN}/monailabel -type f -name "*.so" -delete
  find ${TO_CLEAN}/monailabel -type d -name "__pycache__" -delete
  find ${TO_CLEAN} -maxdepth 1 -type f -name ".coverage.*" -delete

  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".eggs" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "monailabel.egg-info" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "build" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "dist" -exec rm -r "{}" +
  find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".mypy_cache" -exec rm -r "{}" +
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

function print_style_fail_msg() {
  echo "${red}Check failed!${noColor}"
  echo "Please run auto style fixes: ${green}./runtests.sh --autofix${noColor}"
}

function is_pip_installed() {
  return $(${PY_EXE} -c "import sys, pkgutil; sys.exit(0 if pkgutil.find_loader(sys.argv[1]) else 1)" $1)
}

function list_unittests() {
  ${PY_EXE} - <<END
import unittest
def print_suite(suite):
    if hasattr(suite, "__iter__"):
        for x in suite:
            print_suite(x)
    else:
        print(suite)
print_suite(unittest.defaultTestLoader.discover('./tests'))
END
  exit 0
}

if [ -z "$1" ]; then
  print_error_msg "Too few arguments to $0"
  print_usage
fi

# parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --coverage)
    doCoverage=true
    ;;
  -q | --quick)
    doQuickTests=true
    ;;
  --net)
    doNetTests=true
    ;;
  --list_tests)
    list_unittests
    ;;
  --dryrun)
    doDryRun=true
    ;;
  -u | --u*) # allow --unittest | --unittests | --unittesting  etc.
    doUnitTests=true
    ;;
  -f | --codeformat)
    doBlackFormat=true
    doIsortFormat=true
    doFlake8Format=true
    doPytypeFormat=true
    doMypyFormat=true
    ;;
  --black)
    doBlackFormat=true
    ;;
  --autofix)
    doIsortFix=true
    doBlackFix=true
    doIsortFormat=true
    doBlackFormat=true
    ;;
  --isort)
    doIsortFormat=true
    ;;
  --flake8)
    doFlake8Format=true
    ;;
  --pytype)
    doPytypeFormat=true
    ;;
  --mypy)
    doMypyFormat=true
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

if [ $doIsortFormat = true ]; then
  set +e # disable exit on failure so that diagnostics can be given on failure
  if [ $doIsortFix = true ]; then
    echo "${separator}${blue}isort-fix${noColor}"
  else
    echo "${separator}${blue}isort${noColor}"
  fi

  # ensure that the necessary packages for code format testing are installed
  if ! is_pip_installed isort; then
    install_deps
  fi
  ${cmdPrefix}isort --version

  if [ $doIsortFix = true ]; then
    ${cmdPrefix}${PY_EXE} -m isort -l ${LINE_LENGTH} --profile black "$(pwd)"
  else
    ${cmdPrefix}${PY_EXE} -m isort -l ${LINE_LENGTH} --profile black --check "$(pwd)"
  fi

  isort_status=$?
  if [ ${isort_status} -ne 0 ]; then
    print_style_fail_msg
    exit ${isort_status}
  else
    echo "${green}passed!${noColor}"
  fi
  set -e # enable exit on failure
fi

if [ $doBlackFormat = true ]; then
  set +e # disable exit on failure so that diagnostics can be given on failure
  if [ $doBlackFix = true ]; then
    echo "${separator}${blue}black-fix${noColor}"
  else
    echo "${separator}${blue}black${noColor}"
  fi

  # ensure that the necessary packages for code format testing are installed
  if ! is_pip_installed black; then
    install_deps
  fi
  ${cmdPrefix}${PY_EXE} -m black --version

  if [ $doBlackFix = true ]; then
    ${cmdPrefix}${PY_EXE} -m black -l ${LINE_LENGTH} "$(pwd)"
  else
    ${cmdPrefix}${PY_EXE} -m black -l ${LINE_LENGTH} --check "$(pwd)"
  fi

  black_status=$?
  if [ ${black_status} -ne 0 ]; then
    print_style_fail_msg
    exit ${black_status}
  else
    echo "${green}passed!${noColor}"
  fi
  set -e # enable exit on failure
fi

if [ $doFlake8Format = true ]; then
  set +e # disable exit on failure so that diagnostics can be given on failure
  echo "${separator}${blue}flake8${noColor}"

  # ensure that the necessary packages for code format testing are installed
  if ! is_pip_installed flake8; then
    install_deps
  fi
  ${cmdPrefix}${PY_EXE} -m flake8 --version

  ${cmdPrefix}${PY_EXE} -m flake8 "$(pwd)" --count --statistics --max-line-length ${LINE_LENGTH}

  flake8_status=$?
  if [ ${flake8_status} -ne 0 ]; then
    print_style_fail_msg
    exit ${flake8_status}
  else
    echo "${green}passed!${noColor}"
  fi
  set -e # enable exit on failure
fi

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

if [ $doMypyFormat = true ]; then
  set +e # disable exit on failure so that diagnostics can be given on failure
  echo "${separator}${blue}mypy${noColor}"

  # ensure that the necessary packages for code format testing are installed
  if ! is_pip_installed mypy; then
    install_deps
  fi
  ${cmdPrefix}${PY_EXE} -m mypy --version

  if [ $doDryRun = true ]; then
    ${cmdPrefix}MYPYPATH="$(pwd)"/monailabel ${PY_EXE} -m mypy "$(pwd)"
  else
    MYPYPATH="$(pwd)"/monailabel ${PY_EXE} -m mypy "$(pwd)" # cmdPrefix does not work with MYPYPATH
  fi

  mypy_status=$?
  if [ ${mypy_status} -ne 0 ]; then
    : # mypy output already follows format
    exit ${mypy_status}
  else
    : # mypy output already follows format
  fi
  set -e # enable exit on failure
fi

# testing command to run
cmd="${PY_EXE}"

# When running --quick, require doCoverage as well and set QUICKTEST environmental
# variable to disable slow unit tests from running.
if [ $doQuickTests = true ]; then
  echo "${separator}${blue}quick${noColor}"
  doCoverage=true
  export QUICKTEST=True
fi

# set coverage command
if [ $doCoverage = true ]; then
  echo "${separator}${blue}coverage${noColor}"
  cmd="${PY_EXE} -m coverage run --append"
fi

# # download test data if needed
# if [ ! -d testing_data ] && [ "$doDryRun" != 'true' ]
# then
# fi

# unit tests
if [ $doUnitTests = true ]; then
  echo "${separator}${blue}unittests${noColor}"
  torch_validate
  #${cmdPrefix}${cmd} ./tests/runner.py -p "test_((?!integration).)"
  pytest
fi

# network training/inference/eval integration tests
if [ $doNetTests = true ]; then
  echo "${separator}${blue}integration${noColor}"
  for i in tests/*integration_*.py; do
    echo "$i"
    ${cmdPrefix}${cmd} "$i"
  done
fi

# run model zoo tests
if [ $doZooTests = true ]; then
  echo "${separator}${blue}zoo${noColor}"
  print_error_msg "--zoo option not yet implemented"
  exit 255
fi

# report on coverage
if [ $doCoverage = true ]; then
  echo "${separator}${blue}coverage${noColor}"
  ${cmdPrefix}${PY_EXE} -m coverage combine --append .coverage/
  ${cmdPrefix}${PY_EXE} -m coverage report
fi
