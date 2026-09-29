# syntax=docker/dockerfile:1
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

FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages ./packages
COPY scripts/build_package.py ./scripts/build_package.py
RUN python scripts/build_package.py --out-dir /dist

FROM node:22-bookworm-slim AS node
FROM python:3.12-slim-bookworm AS runtime
ARG MONAILABEL_VERSION=1.0.0rc1
LABEL org.opencontainers.image.title="MONAI Label" \
      org.opencontainers.image.source="https://github.com/Project-MONAI/MONAILabel" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version=$MONAILABEL_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg git ffmpeg libglib2.0-0 libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/LICENSE /usr/share/doc/node/LICENSE
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -s ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl --fail --silent --show-error https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli docker-buildx-plugin docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/check_dependencies.py /usr/share/monailabel/check_dependencies.py
RUN --mount=type=bind,from=builder,source=/dist,target=/dist \
    --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --find-links=/dist "monailabel==$MONAILABEL_VERSION" \
    && python /usr/share/monailabel/check_dependencies.py
COPY THIRD_PARTY_NOTICES.md LICENSE /usr/share/monailabel/
RUN python -m pip inspect > /usr/share/monailabel/pip-inspect.json
ENV PYTHONUNBUFFERED=1 \
    MONAILABEL_DATA_DIR=/workspace \
    MONAILABEL_NATIVE_VIEWERS=0 \
    MONAILABEL_DESKTOP_SOCKET_DIR=/tmp/monailabel-sockets
WORKDIR /workspace
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -m monailabel.server.healthcheck
ENTRYPOINT ["/usr/bin/tini", "--", "monailabel"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
