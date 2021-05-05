from setuptools import setup, find_packages
from monailabel import __version__

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_desc = f.read()

with open('requirements.txt') as f:
    requirements = [l.strip() for l in f]

setup(
    name='monai-label',
    version=__version__,
    description='MONAI Label Development Kit',
    long_description=readme,
    url='http://monai.io',
    license=license_desc,
    packages=find_packages(exclude=('tests', 'docs', 'apps')),
    python_requires='>=3.6',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'monai-label = monailabel.main:run_main',
            'monai-label-utils = monailabel.utils.others.app_utils:run_main'
        ]
    },
    scripts=['monailabel/scripts/run_monailabel_user_app.sh'],
    include_package_data=True,
    data_files=[
        ('logconfig', ['monailabel/logging.json']),
        ('userapprequirements', ['monailabel/scripts/user-requirements.txt']),
    ],
)
