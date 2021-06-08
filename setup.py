#!/usr/bin/env python
from setuptools import find_packages, setup

import versioneer

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license_desc = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f]

setup(
    name="monailabel",
    version=versioneer.get_version(),
    description="MONAI Label Development Kit",
    long_description=readme,
    url="http://monai.io",
    license=license_desc,
    packages=find_packages(exclude=("tests", "docs", "apps")),
    python_requires=">=3.6",
    install_requires=requirements,
    scripts=[
        "monailabel/monailabel",
        "monailabel/monailabel.bat",
        "monailabel/scripts/run_monailabel_app.sh",
        "monailabel/scripts/run_monailabel_app.bat",
    ],
    include_package_data=True,
    data_files=[
        ("logconfig", ["monailabel/logging.json"]),
    ],
)
