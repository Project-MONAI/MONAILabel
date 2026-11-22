# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To build with a different base image
# please run `./runtests.sh --clean && DOCKER_BUILDKIT=1 docker build -t projectmonai/monailabel:latest .`
# to use different version of MONAI pass `--build-arg FINAL_IMAGE=...`

#ARG FINAL_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# monailabel needs python >= 3.11 (ubuntu:22.04 ships 3.10)
ARG FINAL_IMAGE=ubuntu:24.04
ARG BUILD_IMAGE=python:3.11
# OHIF builds with yarn 1.x, which the node images stopped bundling after node 24
ARG NODE_IMAGE=node:22-slim

# Phase1: Build OHIF Viewer
FROM ${NODE_IMAGE} AS ohifbuild
COPY plugins/ohifv3 /opt/ohifv3
RUN apt update -y && apt install -y git
RUN cd /opt/ohifv3 && ./build.sh /opt/ohifv3/release

# Phase2: Build MONAI Label Package
FROM ${BUILD_IMAGE} AS build
WORKDIR /opt/monailabel
RUN python -m pip install pip setuptools wheel twine
# setup.cfg lists torch and ninja in setup_requires; install them with pip (CPU build is enough here)
# because setuptools can no longer fetch them itself
RUN python -m pip install --no-cache-dir ninja torch --extra-index-url https://download.pytorch.org/whl/cpu
ADD . /opt/monailabel/
COPY --from=ohifbuild /opt/ohifv3/release /opt/monailabel/monailabel/endpoints/static/ohif
RUN BUILD_OHIF=false python setup.py bdist_wheel --build-number $(date +'%Y%m%d%H%M')

# Phase3: Build Final Docker
FROM ${FINAL_IMAGE}
LABEL maintainer="monai.contact@gmail.com"
WORKDIR /opt/monailabel
COPY requirements.txt /opt/monailabel/requirements.txt

RUN apt update -y && apt install -y git curl openslide-tools python3 python-is-python3 python3-pip python3-venv

# install into a virtual environment (the system python is externally managed, PEP 668) with a current pip;
# packages that FINAL_IMAGE already provides stay visible
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir pytest torch torchvision torchaudio

COPY --from=build /opt/monailabel/dist/monailabel* /opt/monailabel/dist/
RUN python -m pip install -v --no-cache-dir /opt/monailabel/dist/monailabel*.whl
RUN python -m pip uninstall sam2 -y
RUN python -m pip install -v --no-cache-dir -r /opt/monailabel/requirements.txt
