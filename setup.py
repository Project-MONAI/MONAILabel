from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_desc = f.read()

setup(
    name='monai-label',
    version='1.0.0',
    description='MONAI Label',
    long_description=readme,
    author='MONAI Label',
    author_email='label@monai.com',
    url='http://monai.com',
    license=license_desc,
    packages=find_packages(exclude=('tests', 'docs', 'apps'))
)
