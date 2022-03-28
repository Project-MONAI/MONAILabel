FROM python:3.9-slim

RUN python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels

ADD . /opt/monailabel/dsa
WORKDIR /opt/monailabel/dsa/cli

ENV PYTHONPATH=$PYTHONPATH:/opt/monailabel/dsa

RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint MONAILabelAnnotation --help
RUN python -m slicer_cli_web.cli_list_entrypoint MONAILabelTraining --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
