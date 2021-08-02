#!/usr/bin/env python
import os
import platform
from distutils.command.sdist import sdist
from distutils.errors import DistutilsExecError

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

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


class OHIFSDist(sdist):
    def run(self):
        try:
            script = "build.bat" if any(platform.win32_ver()) else "build.sh"
            command = os.path.realpath(os.path.join(os.path.dirname(__file__), "plugins", "ohif", script))
            self.spawn(['sh', command])
        except DistutilsExecError:
            self.warn('listing directory failed')
        super().run()


class OHIFBuildPy(build_py):
    def run(self):
        script = "build.bat" if any(platform.win32_ver()) else "build.sh"
        command = os.path.realpath(os.path.join(os.path.dirname(__file__), "plugins", "ohif", script))
        self.spawn(['sh', command])
        super().run()


setup(
    version=versioneer.get_version(),
    cmdclass={
        'sdist': OHIFSDist,
        'build_py': OHIFBuildPy,
    },
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
