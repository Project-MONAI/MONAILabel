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
# to exclude ORTHANC pass `--build-arg ORTHANC=false`

ARG MONAI_IMAGE=projectmonai/monai:latest
ARG ORTHANC=false

FROM ${MONAI_IMAGE} as build
LABEL maintainer="monai.contact@gmail.com"

ADD . /opt/monailabel/
RUN apt update -y && apt install openslide-tools npm -y && npm install --global yarn
RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel twine \
    && cd /opt/monailabel \
    && python setup.py sdist bdist_wheel --build-number $(date +'%Y%m%d%H%M')

FROM ${MONAI_IMAGE}
LABEL maintainer="monai.contact@gmail.com"

COPY --from=build /opt/monailabel/dist/monailabel* /opt/monailabel/dist/
RUN python -m pip install --upgrade --no-cache-dir pip \
    && python -m pip install /opt/monailabel/dist/monailabel*.whl
RUN python3 -m pip install numpy --upgrade

# Add Orthanc
RUN if [ "${ORTHANC}" = "true" ] ; then  \
    apt-get update -y && apt-get install orthanc orthanc-dicomweb plastimatch -y \
    && service orthanc stop \
    && wget https://lsb.orthanc-server.com/orthanc/1.9.6/Orthanc --output-document /usr/sbin/Orthanc \
    && rm -f /usr/share/orthanc/plugins/*.so \
    && wget https://lsb.orthanc-server.com/orthanc/1.9.6/libServeFolders.so --output-document /usr/share/orthanc/plugins/libServeFolders.so \
    && wget https://lsb.orthanc-server.com/orthanc/1.9.6/libModalityWorklists.so --output-document /usr/share/orthanc/plugins/libModalityWorklists.so \
    && wget https://lsb.orthanc-server.com/plugin-dicom-web/1.6/libOrthancDicomWeb.so --output-document /usr/share/orthanc/plugins/libOrthancDicomWeb.so \
    && service orthanc restart; fi
