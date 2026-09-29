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

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 tigervnc-standalone-server xauth x11-utils openbox \
    libglu1-mesa libopengl0 libgl1-mesa-dri libpulse-mainloop-glib0 libnss3 \
    libsm6 libxcb-cursor0 libxkbcommon-x11-0 libxi6 libxrender1 libxtst6 \
    libasound2t64 libgtk-3-0t64 fonts-dejavu-core qt5dxcb-plugin \
    && if [ "$(dpkg --print-architecture)" = arm64 ]; then \
       apt-get install -y --no-install-recommends \
       libqt5xml5 libqt5xmlpatterns5 libqt5multimediawidgets5 libqt5svg5 \
       libqt5opengl5 libqt5sql5-sqlite libqt5webengine5 libqt5webenginewidgets5 \
       libqt5designer5; fi \
    && rm -rf /var/lib/apt/lists/*
COPY entrypoint.py /opt/desktop.py
COPY rc.xml /etc/monailabel-openbox.xml
ENV HOME=/session DISPLAY=:1 LIBGL_ALWAYS_SOFTWARE=1 \
    MONAILABEL_VIEWER_SESSION=/session/launch.json \
    MONAILABEL_QUPATH_DATA=/session/projects \
    JAVA_TOOL_OPTIONS="-Duser.home=/session -Djava.util.prefs.userRoot=/session/preferences -Djava.io.tmpdir=/session/tmp"
ENTRYPOINT ["python3", "/opt/desktop.py"]
