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

import distutils.util
import hashlib
import json
import logging
import mimetypes
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict
from urllib.parse import unquote, urlparse

import torch
from monai.apps import download_url
from monai.bundle import download, get_bundle_versions
from monai.bundle.scripts import get_all_bundles_list

from monailabel.config import settings
from monailabel.utils.others.modelzoo_list import MAINTAINED_BUNDLES

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


def get_basename_no_ext(path):
    p = get_basename(path)
    e = file_ext(p)
    return p.rstrip(e)


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
        log_config = tempfile.NamedTemporaryFile(suffix=".json").name
        os.makedirs(log_dir, exist_ok=True)

        # if not os.path.exists(log_config):
        shutil.copyfile(default_config, log_config)
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
    if torch.cuda.device_count() == 1:
        devices.append(torch.cuda.get_device_name(0))
    else:
        for i in range(torch.cuda.device_count()):
            devices.append(f"{torch.cuda.get_device_name(i)}:{i}")

    return devices


def device_map():
    devices = {} if torch.cuda.is_available() else {"cpu": "cpu"}
    if torch.cuda.device_count() == 1:
        devices[torch.cuda.get_device_name(0)] = "cuda"
    else:
        for i in range(torch.cuda.device_count()):
            devices[f"{torch.cuda.get_device_name(i)}:{i}"] = f"cuda:{i}"

    return devices


def name_to_device(device):
    device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
    device = device if isinstance(device, str) else device[0]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return device_map().get(device, device)


def create_dataset_from_path(folder, image_dir="images", label_dir="labels", img_ext=".jpg", lab_ext=".png"):
    def _list_files(d, ext):
        files = [i for i in os.listdir(d) if i.endswith(ext)]
        return sorted(os.path.join(d, i) for i in files)

    image_dir = os.path.join(folder, image_dir) if image_dir else folder
    images = _list_files(image_dir, img_ext)

    label_dir = os.path.join(folder, label_dir) if label_dir else folder
    labels = _list_files(label_dir, lab_ext)

    for i, l in zip(images, labels):
        if get_basename_no_ext(i) != get_basename_no_ext(l):
            logger.warning(f"NO MATCH: {i} => {l}")

    return [
        {"image": i, "label": l} for i, l in zip(images, labels) if get_basename_no_ext(i) == get_basename_no_ext(l)
    ]


def strtobool(s):
    return False if s is None else s if isinstance(s, bool) else bool(distutils.util.strtobool(s))


def is_openslide_supported(name):
    ext = file_ext(name)
    supported_ext = (".bif", ".mrxs", ".ndpi", ".scn", ".svs", ".svslide", ".tif", ".tiff", ".vms", ".vmu")
    if ext and ext in supported_ext:
        return True
    return False


def get_zoo_bundle(model_dir, conf, models, conf_key):
    zoo_repo = conf.get("zoo_repo", settings.MONAI_ZOO_REPO)
    zoo_info = get_all_bundles_list()

    # filter model zoo bundle with MONAI Label supported bundles according to the maintaining list, return all version bundles list
    available = [i[0] for i in zoo_info if i[0] in MAINTAINED_BUNDLES]
    available_with_version = {b: get_bundle_versions(b)["all_versions"] for b in available}

    available_both = available + [k + "_v" + v for k, versions in available_with_version.items() for v in versions]

    version_to_name = {k + "_v" + v: k for k, versions in available_with_version.items() for v in versions}
    name_to_version = {k + "_v" + v: v for k, versions in available_with_version.items() for v in versions}
    if not models:
        print("")
        print("---------------------------------------------------------------------------------------")
        print("Get models from bundle configs, Please provide --conf models <bundle name>")
        print("Following are the available bundles.  You can pass comma (,) separated names to pass multiple")
        print(f"   -c {conf_key}")
        print("        {}".format(" \n        ".join(available)))
        print("---------------------------------------------------------------------------------------")
        print("")
        exit(-1)

    # First check whether the bundle model directory is available and in model-zoo, if no, check local bundle directory.
    # Use zoo bundle if both exist
    invalid_zoo = [m for m in models if m not in available_both]

    invalid = [m for m in invalid_zoo if not os.path.isdir(os.path.join(model_dir, m))]
    if invalid:
        print("")
        print("---------------------------------------------------------------------------------------")
        print(f"Invalid Model(s) are provided: {invalid}, either not in model zoo or not supported with MONAI Label")
        print("Following are the available models.  You can pass comma (,) separated names to pass multiple")
        print("Available bundle with latest tags:")
        print(f"   -c {conf_key}")
        print("        {}".format(" \n        ".join(available)))
        print("Or provide valid local bundle directories")
        print("---------------------------------------------------------------------------------------")
        print("")
        exit(-1)

    bundles: Dict[str, str] = {}

    for k in models:
        p = os.path.join(model_dir, k)
        if k not in available_both:
            logger.info(f"+++ Adding Bundle from Local: {k} => {p}")
        else:
            logger.info(f"+++ Adding Bundle from Zoo: {k} => {p}")
            if not os.path.exists(p):
                name = k if k in available else version_to_name.get(k)
                version = None if k in available else name_to_version.get(k)
                download(name=name, version=version, bundle_dir=model_dir, source="monaihosting", repo=zoo_repo)
                if version:
                    shutil.move(os.path.join(model_dir, name), p)
        bundles[k] = p

    return bundles


def get_bundle_models(app_dir, conf, conf_key="models"):
    """
    The funtion to get bundle models either from available model zoo or local files.
    MONAI Label maintains a list of supported bundles, non-labeling bundles are not supported.
    This function will filter available bundles according to the maintaining list.

    Args:
        app_dir: the app directory path
        conf: configs of start_server command
        conf_key: default to "models" for monaibundle app, if radiology app wants to use bundle models, "--conf bundles <names>" is used.

    Returns:
        a dictionary that contains the available bundles.

    Example:
        Bundle name: spleen_ct_segmentation, this will use latest version of the bundle.
        BUndle name with specific version: spleen_ct_segmentation_v0.4.0, this will download the specified version.
    """
    model_dir = os.path.join(app_dir, "model")

    zoo_source = conf.get("zoo_source", settings.MONAI_ZOO_SOURCE)

    models = conf.get(conf_key)
    models = models.split(",") if models else []
    models = [m.strip() for m in models]

    if zoo_source == "monaihosting":  # if in github env, access model zoo
        bundles = get_zoo_bundle(model_dir, conf, models, conf_key)
    else:  # if not in github env, no "model zoo" access, users either provide bundles locally, or auto download with latest
        bundles: Dict[str, str] = {}
        for k in models:
            p = os.path.join(model_dir, k)
            if os.path.isdir(p):
                logger.info(f"+++ Adding Bundle from Local: {k} => {p}")
            else:
                logger.info(f"+++ Adding Bundle from NGC: {k} => {p}")
                pattern = re.compile(r"(?P<name>.+)_v(?P<version>\d+\.\d+\.\d+)")
                match = pattern.match(k)
                if match:
                    name = match.group("name")
                    version = match.group("version") or None
                else:
                    name = k
                    version = None
                download(name=name, version=version, bundle_dir=model_dir, source=zoo_source)

            bundles[k] = p

    logger.info(f"+++ Using Bundle Models: {list(bundles.keys())}")
    return bundles


def path_to_uri(path) -> str:
    return pathlib.Path(path).absolute().as_uri()


def path_to_sqlite_uri(path) -> str:
    """SQLAlchemy URI of a SQLite database file, e.g. for MLflow (its file store backend is no longer supported)."""
    return "sqlite:///" + pathlib.Path(path).absolute().as_posix()


def is_plain_filename(name) -> bool:
    """True if ``name`` is a file name only: no directory part, so joining it to a folder stays inside that folder."""
    # ":" covers Windows drive-relative names ("D:model.pt" leaves the folder it is joined to) and NTFS streams
    return isinstance(name, str) and name not in ("", ".", "..") and not re.search(r"[/\\:\x00]", name)


def is_googleapis_url(url) -> bool:
    """True if the host of ``url`` is googleapis.com or one of its sub domains (e.g. healthcare.googleapis.com)."""
    host = (urlparse(url or "").hostname or "").lower()
    return host == "googleapis.com" or host.endswith(".googleapis.com")


def has_path_traversal(path: str) -> bool:
    """True if a URL path (possibly percent-encoded more than once) has a ``.`` or ``..`` segment or a NUL character."""
    for _ in range(4):  # a client can encode more than once; stop when decoding no longer changes anything
        decoded = unquote(path)
        if decoded == path:
            break
        path = decoded
    return "\x00" in path or any(segment in (".", "..") for segment in re.split(r"[/\\]", path))


class UnsafeRequestError(ValueError):
    """A client-supplied value or file is rejected because of what the server would do with it (HTTP 400)."""


class UnsafeConfigValueError(UnsafeRequestError):
    """A client-supplied value would be interpreted as MONAI bundle config syntax instead of plain data."""


class UnsafeImageFileError(UnsafeRequestError):
    """An image file whose header makes the reader load the pixel data from another file on the server."""


def int_in_range(value, low: int, high: int, name: str = "value") -> int:
    """Return ``value`` as int if it is a whole number in ``[low, high]``; client numbers must not be anything else."""
    try:
        whole = not isinstance(value, bool) and isinstance(value, (int, float)) and int(value) == value
    except (OverflowError, ValueError):  # inf, nan
        whole = False
    if not whole:
        raise UnsafeRequestError(f"Invalid value for '{name}': a whole number is expected")
    if not low <= value <= high:
        raise UnsafeRequestError(f"Invalid value for '{name}': it must be between {low} and {high}")
    return int(value)


_HEADER_LIMIT = 1024 * 1024
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_NRRD_DATA_FILE = re.compile(rb"^[ \t]*data[ \t]*file[ \t]*:", re.IGNORECASE | re.MULTILINE)
_META_DATA_FILE = re.compile(rb"^[ \t]*ElementDataFile[ \t]*=[ \t]*([^\r\n]*)", re.IGNORECASE | re.MULTILINE)


def check_image_file(path, name: str = "image"):
    """Return ``path`` unchanged unless it is an image file that points the reader at another file.

    NRRD (``data file:``), MetaImage (``ElementDataFile``) and HDF5/MINC2 (external storage) headers can name any
    file on the server as the pixel data; such an upload would turn the image loader into a file reader.
    Anything that is not an existing regular file (ids, directories, non-string inputs) is returned as is.

    Raises:
        UnsafeImageFileError: for a detached-data header, or a header too large to be checked.
    """
    if not isinstance(path, str) or not os.path.isfile(path):
        return path

    with open(path, "rb") as fp:
        head = fp.read(_HEADER_LIMIT)
        offset = 512
        hdf5 = head.startswith(_HDF5_SIGNATURE)
        while not hdf5 and offset < len(head):  # the signature may follow a user block of 512, 1024, 2048 ... bytes
            hdf5 = head[offset : offset + len(_HDF5_SIGNATURE)] == _HDF5_SIGNATURE
            offset *= 2
    if hdf5:
        raise UnsafeImageFileError(f"Invalid {name}: HDF5 based image files are not accepted")

    lower = path.lower()
    if head.startswith(b"NRRD") or lower.endswith((".nrrd", ".nhdr")):
        ends = [i for i in (head.find(b"\n\n"), head.find(b"\r\n\r\n")) if i >= 0]
        if not ends and len(head) >= _HEADER_LIMIT:
            raise UnsafeImageFileError(f"Invalid {name}: NRRD header is too large")
        header = head[: min(ends)] if ends else head
        if _NRRD_DATA_FILE.search(header):
            raise UnsafeImageFileError(f"Invalid {name}: NRRD files with a detached data file are not accepted")

    meta = _META_DATA_FILE.search(head)
    if lower.endswith((".mha", ".mhd")) and not meta:
        raise UnsafeImageFileError(f"Invalid {name}: MetaImage header has no ElementDataFile entry")
    if meta and meta.group(1).strip().upper() != b"LOCAL":
        raise UnsafeImageFileError(f"Invalid {name}: MetaImage files with a detached data file are not accepted")
    return path


# MONAI's ConfigParser treats a string starting with "$" as a Python expression (it is eval()'ed), one starting with
# "%" as a macro that is replaced by the content of another (any) config file, and a string that is exactly "@id"
# as a reference. A mapping with a "_target_" key is instantiated (it can name any importable callable); the other
# "_xxx_" keys (_args_, _requires_, _mode_ ...) are reserved for the same machinery, "+" merges into an existing entry.
_CONFIG_STRING_PREFIXES = ("$", "@", "%")
_CONFIG_KEY_PREFIXES = ("_", "+") + _CONFIG_STRING_PREFIXES


def check_config_value(value, name: str = "value"):
    """Return ``value`` unchanged if it is safe to put into a MONAI bundle config, else raise.

    Values received from a client (request parameters) must be plain data. Anything that ``ConfigParser`` would
    evaluate, resolve or instantiate (see above) would run attacker-chosen code in the server process, so strings,
    mapping keys and nested list/dict content are checked recursively.

    Raises:
        UnsafeConfigValueError: for a string/key that starts with ``$``, ``@`` or ``%``, or a mapping key
            starting with ``_`` or ``+``. ``name`` is used in the message (the value itself is not echoed).
    """
    if isinstance(value, str):
        if value.lstrip().startswith(_CONFIG_STRING_PREFIXES):
            raise UnsafeConfigValueError(f"Invalid value for '{name}': it must not start with '$', '@' or '%'")
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and k.lstrip().startswith(_CONFIG_KEY_PREFIXES):
                raise UnsafeConfigValueError(f"Invalid key in '{name}': it must not start with '_', '+', '$', '@', '%'")
            check_config_value(v, name)
    elif isinstance(value, (list, tuple)):
        for v in value:
            check_config_value(v, name)
    return value


def handle_torch_linalg_multithread(req):
    try:
        if torch.cuda.is_available():
            torch.inverse(torch.eye(1, device=req.get("device") if req else None))
    except RuntimeError:
        pass


def md5_digest(s: str) -> str:
    if sys.version_info.minor < 9:
        return hashlib.md5(s.encode("utf-8")).hexdigest()
    return hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()
