#!/usr/bin/env python
import os

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
