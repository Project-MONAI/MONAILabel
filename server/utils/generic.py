import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def run_command(command, args=None, plogger=None):
    plogger = plogger if plogger else logger
    cmd = [command]
    if args:
        args = [str(a) for a in args]
        cmd.extend(args)

    plogger.info('Running Command:: {}'.format(' '.join(cmd)))
    process = subprocess.Popen(
        cmd,
        # stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        env=os.environ.copy()
    )

    while process.poll() is None:
        line = process.stdout.readline()
        if line:
            plogger.info(line.rstrip())

    plogger.info('Return code: {}'.format(process.returncode))
    process.stdout.close()
    return process.returncode


def init_log_config(log_config, app_dir, log_file):
    if not log_config or not os.path.exists(log_config):
        default_config = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'logging.json'))
        log_dir = os.path.join(app_dir, "logs")
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
