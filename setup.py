#!/usr/bin/env python
import distutils
import os
import platform
import subprocess

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
build_ohif = os.environ.get("BUILD_OHIF")
print(f"BUILD_OHIF = {build_ohif}")
build_ohif = True if not build_ohif else distutils.util.strtobool(build_ohif)
if build_ohif:
    script = "build.bat" if any(platform.win32_ver()) else "build.sh"
    command = os.path.realpath(os.path.join(os.path.dirname(__file__), "plugins", "ohif", script))
    if os.path.exists(command):
        subprocess.call(["sh", command])

setup(
    version=versioneer.get_version(),
    packages=find_packages(exclude=("tests", "docs", "sample-apps", "plugins")),
    zip_safe=False,
    package_data={"monailabel": ["py.typed"]},
    include_package_data=True,
    scripts=[
        "monailabel/monailabel",
        "monailabel/monailabel.bat",
        "monailabel/scripts/run_monailabel_app.sh",
        "monailabel/scripts/run_monailabel_app.bat",
    ],
    data_files=data_files,
)
