# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import logging
import mimetypes
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)


def run_command(command, args=None, plogger=None):
    plogger = plogger if plogger else logger
    cmd = [command]
    if args:
        args = [str(a) for a in args]
        cmd.extend(args)

    plogger.info("Running Command:: {}".format(" ".join(cmd)))
    process = subprocess.Popen(
        cmd,
        # stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        env=os.environ.copy(),
    )

    while process.poll() is None:
        line = process.stdout.readline()
        line = line.rstrip()
        if line:
            plogger.info(line.rstrip()) if plogger else print(line)

    plogger.info("Return code: {}".format(process.returncode))
    process.stdout.close()
    return process.returncode


def init_log_config(log_config, app_dir, log_file):
    if not log_config or not os.path.exists(log_config):
        default_log_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        default_config = os.path.realpath(os.path.join(default_log_dir, "logging.json"))

        log_dir = os.path.join(app_dir, "logs")
        log_config = os.path.join(log_dir, "logging.json")
        os.makedirs(log_dir, exist_ok=True)

        # if not os.path.exists(log_config):
        shutil.copy(default_config, log_config)
        with open(log_config, "r") as f:
            c = f.read()

        c = c.replace("${LOGDIR}", log_dir.replace("\\", r"\\"))
        c = c.replace("${LOGFILE}", os.path.join(log_dir, log_file).replace("\\", r"\\"))

        with open(log_config, "w") as f:
            f.write(c)

    return log_config


def get_mime_type(file):
    m_type = mimetypes.guess_type(file, strict=False)
    logger.debug(f"Guessed Mime Type for Image: {m_type}")

    if m_type is None or m_type[0] is None:
        m_type = "application/octet-stream"
    else:
        m_type = m_type[0]
    logger.debug(f"Final Mime Type: {m_type}")
    return m_type


def file_checksum(file, algo="SHA256"):
    if algo not in ["SHA256", "SHA512", "MD5"]:
        raise ValueError("unsupported hashing algorithm %s" % algo)

    with open(file, "rb") as content:
        hash = hashlib.new(algo)
        while True:
            chunk = content.read(8192)
            if not chunk:
                break
            hash.update(chunk)
        return hash.hexdigest()
