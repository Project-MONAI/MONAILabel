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
# to use different version of MONAI pass `--build-arg MONAI_IMAGE=...`

ARG MONAI_IMAGE=projectmonai/monai:1.4.0
ARG BUILD_IMAGE=python:3.10
ARG NODE_IMAGE=node:slim

# Phase1: Build OHIF Viewer
FROM ${NODE_IMAGE} AS ohifbuild
COPY plugins/ohifv3 /opt/ohifv3
RUN apt update -y && apt install -y git
RUN cd /opt/ohifv3 && ./build.sh /opt/ohifv3/release

# Phase2: Build MONAI Label Package
FROM ${BUILD_IMAGE} AS build
ADD . /opt/monailabel/
COPY --from=ohifbuild /opt/ohifv3/release /opt/monailabel/monailabel/endpoints/static/ohif
RUN python -m pip install pip setuptools wheel twine
RUN cd /opt/monailabel && BUILD_OHIF=false python setup.py sdist bdist_wheel --build-number $(date +'%Y%m%d%H%M')

# Phase3: Build Final Docker based on MONAI
FROM ${MONAI_IMAGE}
LABEL maintainer="monai.contact@gmail.com"
COPY --from=build /opt/monailabel/dist/monailabel* /opt/monailabel/dist/
RUN SAM2_BUILD_CUDA=0 python -m pip install /opt/monailabel/dist/monailabel*.whl
