import logging
import os
import shlex
import shutil
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


def init_log_config(log_config, workspace, log_file):
    if not log_config or not os.path.exists(log_config):
        default_config = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'logging.json'))
        log_dir = os.path.join(workspace, "logs")
        log_config = os.path.join(log_dir, "logging.json")
        os.makedirs(log_dir, exist_ok=True)

        # if not os.path.exists(log_config):
        shutil.copy(default_config, log_config)
        with open(log_config, 'r') as f:
            c = f.read()

        c = c.replace("${LOGDIR}", log_dir)
        c = c.replace("${LOGFILE}", log_file)

        with open(log_config, 'w') as f:
            f.write(c)

    return log_config
