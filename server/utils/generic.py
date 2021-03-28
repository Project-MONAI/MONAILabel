import logging
import os
import shlex
import subprocess

logger = logging.getLogger(__name__)


def run_command(command, log_prefix='CMD:: '):
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        universal_newlines=True,
        env=os.environ.copy()
    )
    logger.info('Running Command: {}'.format(command))

    while process.poll() is None:
        line = process.stdout.readline()
        if line:
            logger.info('{}{}'.format(log_prefix, line.rstrip()))

    # with io.TextIOWrapper(process.stdout, encoding="utf-8") as f:
    #    for line in f:
    #        logger.info('{}{}'.format(log_prefix, line.rstrip()))

    logger.info('{}Return code: {}'.format(log_prefix, process.returncode))
    process.stdout.close()
    return process.returncode
