FROM ubuntu@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential cmake ninja-build ca-certificates curl patch \
    qtmultimedia5-dev qttools5-dev libqt5xmlpatterns5-dev libqt5svg5-dev \
    qtwebengine5-dev qtscript5-dev qtbase5-private-dev libqt5x11extras5-dev \
    libxt-dev libssl-dev libglu1-mesa-dev libpulse-dev libnss3-dev \
    libxrender-dev libxext-dev libxkbcommon-x11-dev libxcb-cursor0 \
    libasound2-dev libsm-dev libice-dev libxi-dev libxtst-dev libqt5opengl5-dev \
    && rm -rf /var/lib/apt/lists/*
