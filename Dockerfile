ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:21.02-py3
FROM ${PYTORCH_IMAGE}

LABEL maintainer="monai.contact@gmail.com"

WORKDIR /opt/monai_label

# install full deps
COPY requirements.txt /tmp/
RUN python -m pip install --upgrade --no-cache-dir pip \
  && python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/monai_label
ENV PATH=$PATH:"/opt/monai_label"
