ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:21.02-py3
FROM ${PYTORCH_IMAGE}

LABEL maintainer="monai.contact@gmail.com"

WORKDIR /opt/monailabel

# install full deps
COPY requirements.txt /tmp/
RUN python -m pip install --upgrade --no-cache-dir pip \
  && python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/monailabel
ENV PATH=$PATH:"/opt/monailabel"
