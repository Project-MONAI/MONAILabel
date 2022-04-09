:: Copyright (c) MONAI Consortium
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::     http://www.apache.org/licenses/LICENSE-2.0
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

@echo off

set app_dir=%1
set study_dir=%2
set method=%3
set request=%4

set PY_EXE=python

pip install mypy

echo "Clean up (Need powershell installed)"
powershell -Command "& {rm -recurse -force sample-apps/*/logs}"
powershell -Command "& {rm -recurse -force sample-apps/*/.venv}"
powershell -Command "& {rm -recurse -force sample-apps/*/model}"
powershell -Command "& {rm -recurse -force tests/data/*}"
git restore sample-apps/*/model/.gitignore
git restore tests/data/.gitignore

echo "Running Auto Check..."
%PY_EXE% -m mypy .
