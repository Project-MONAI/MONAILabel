# DSA MONAILabel extension

## Requirement

You need to install DSA and upload some test images.
> Refer: https://github.com/DigitalSlideArchive/digital_slide_archive/tree/master/devops/dsa

## Installation

You can use [projectmonai/monailabel-dsa:latest](https://hub.docker.com/r/projectmonai/monailabel-dsa/tags) plugin from
dockerhub. Otherwise you can build from sources as follows.

```bash
  cd plugins/dsa
  docker build -t projectmonai/monailabel-dsa:latest .
```

You can pass default MONAILabel server by defining following environment variable
in [docker compose](https://github.com/DigitalSlideArchive/digital_slide_archive/blob/master/devops/dsa/docker-compose.yml)
for DSA.
This will also help to pre-fetch the available models from MONAILabel server.

```
---
version: '3'
services:
  girder:
    environment:
      SLICER_CLI_WEB_MONAI_LABEL_SERVER: http://monai.label.com:8020
```

Upload new Task (under Slicer CLI Web Tasks) using the above docker image. This will add/enable MONAILabel under
Analysis Page.

![image](../../docs/images/dsa.jpg)


> **NOTE::** Provide
> additional [options](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/pathology#digital-slide-arhive-dsa-as-datastore)
> like `dsa_api_key` and `dsa_asset_store_path` while running the MONAILabel server for faster resource sharing between
> MONAILabel and DSA.