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

import hashlib
import json
import logging
import mimetypes
import os
import pathlib
import shutil
import subprocess
import time

import torch
from monai.apps import download_url

logger = logging.getLogger(__name__)


def file_ext(name) -> str:
    suffixes = []
    for s in reversed(pathlib.Path(name).suffixes):
        if len(s) > 10:
            break
        suffixes.append(s)
    return "".join(reversed(suffixes)) if name else ""


def remove_file(path: str) -> None:
    if path and os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


def get_basename(path):
    """Gets the basename of a file.

    Ref: https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    """
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)


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

    plogger.info(f"Return code: {process.returncode}")
    process.stdout.close()
    return process.returncode


def init_log_config(log_config, app_dir, log_file, root_level=None):
    if not log_config or not os.path.exists(log_config):
        default_log_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        default_config = os.path.realpath(os.path.join(default_log_dir, "logging.json"))

        log_dir = os.path.join(app_dir, "logs")
        log_config = os.path.join(log_dir, "logging.json")
        os.makedirs(log_dir, exist_ok=True)

        # if not os.path.exists(log_config):
        shutil.copy(default_config, log_config)
        with open(log_config) as f:
            c = f.read()

        c = c.replace("${LOGDIR}", log_dir.replace("\\", r"\\"))
        c = c.replace("${LOGFILE}", os.path.join(log_dir, log_file).replace("\\", r"\\"))

        with open(log_config, "w") as f:
            f.write(c)

    with open(log_config) as f:
        j = json.load(f)

    if root_level and j["root"]["level"] != root_level:
        j["root"]["level"] = root_level
        with open(log_config, "w") as f:
            json.dump(j, f, indent=2)

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
        return f"{algo}:{hash.hexdigest()}"


def gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    logger.info("Using nvidia-smi command")
    if shutil.which("nvidia-smi") is None:
        logger.info("nvidia-smi command didn't work! - Using default image size [128, 128, 64]")
        return {0: 4300}

    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"], encoding="utf-8"
    )

    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def gpu_count():
    return torch.cuda.device_count()


def download_file(url, path, delay=1, skip_on_exists=True):
    if skip_on_exists and os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Downloading resource: {path} from {url}")

    download_url(url, path)
    if delay > 0:
        time.sleep(delay)


def device_list():
    devices = [] if torch.cuda.is_available() else ["cpu"]
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")

    return devices
