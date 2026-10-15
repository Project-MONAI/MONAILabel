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

# Offline workflow test only: reuse the dependencies of an installed desktop image.
# Release builds must still validate the production Dockerfile independently.
ARG DESKTOP_BASE_IMAGE
FROM ${DESKTOP_BASE_IMAGE}
COPY --chmod=0444 entrypoint.py /opt/desktop.py
COPY --chmod=0444 relay.py /opt/desktop-relay.py
COPY --chmod=0444 rc.xml /etc/monailabel-openbox.xml
