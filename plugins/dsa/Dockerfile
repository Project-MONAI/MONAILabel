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

FROM python:3.9-slim

RUN python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels

ADD . /opt/monailabel/dsa
WORKDIR /opt/monailabel/dsa/cli

ENV PYTHONPATH=$PYTHONPATH:/opt/monailabel/dsa

RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint MONAILabelAnnotation --help
RUN python -m slicer_cli_web.cli_list_entrypoint MONAILabelTraining --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
