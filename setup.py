#!/usr/bin/env python
from setuptools import find_packages, setup

import versioneer

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
    data_files=[
        ("logconfig", ["monailabel/logging.json"]),
    ],
)
