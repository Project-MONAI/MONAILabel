#!/usr/bin/env python

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

import os
import platform
import subprocess
from distutils.util import strtobool

from setuptools import find_packages, setup

import versioneer


def recursive_files(directory, prefix):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        filenames = [f for f in filenames if f != ".gitignore"]
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return [(os.path.dirname(prefix + os.path.sep + p), [p]) for p in paths]


data_files = [("logconfig", ["monailabel/logging.json"])]
data_files.extend(recursive_files("sample-apps", "monailabel"))
data_files.extend(recursive_files("plugins/slicer", "monailabel"))

# Build OHIF Plugin
build_ohif_s = os.environ.get("BUILD_OHIF", "true")
print(f"BUILD_OHIF = {build_ohif_s}")
build_ohif = True if not build_ohif_s else strtobool(build_ohif_s)
if build_ohif:
    script = "build.bat" if any(platform.win32_ver()) else "build.sh"
    command = os.path.realpath(os.path.join(os.path.dirname(__file__), "plugins", "ohif", script))
    if os.path.exists(command):
        subprocess.call(["sh", command])


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(exclude=("tests", "docs", "sample-apps", "plugins")),
    zip_safe=False,
    package_data={"monailabel": ["py.typed"]},
    include_package_data=True,
    scripts=[
        "monailabel/scripts/monailabel",
        "monailabel/scripts/monailabel.bat",
    ],
    data_files=data_files,
)
