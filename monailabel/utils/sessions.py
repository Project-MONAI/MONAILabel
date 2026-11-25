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

import json
import logging
import os
import pathlib
import re
import shutil
import tarfile
import tempfile
import time
import uuid
import zipfile
import zlib
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _is_inside(path: str, base: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), base]) == base
    except ValueError:
        # e.g. Windows paths on different drives have no common path
        return False


def safe_unpack_archive(archive: str, extract_dir: str) -> None:
    """Extract a zip/tar archive, requiring every member to stay inside ``extract_dir``.

    Tar member names are written as-is by :mod:`tarfile` on Python < 3.14, so each one is
    resolved against the extraction directory before anything is written, and links and
    special files (devices, fifos) are not extracted. :mod:`zipfile` already drops absolute
    and ``..`` components from member names. Raises :class:`ValueError` for a rejected
    member, a corrupt archive or an unsupported format.
    """
    base = os.path.realpath(extract_dir)
    name = os.path.basename(archive)

    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(base)
            return

        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                members = tf.getmembers()
                for member in members:
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(f"Invalid archive: member {member.name!r} is a link or special file")
                    if os.path.isabs(member.name) or not _is_inside(os.path.join(base, member.name), base):
                        raise ValueError(f"Invalid archive: member {member.name!r} is outside the extraction directory")
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(base, members=members, filter="data")
                else:
                    tf.extractall(base, members=members)
            return
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, zlib.error) as e:
        # EOFError / zlib.error: truncated or corrupt compressed stream
        raise ValueError(f"Invalid archive: {name!r} could not be extracted ({e})") from e

    raise ValueError(f"Invalid archive: {name!r} is not a supported zip or tar file")


class SessionInfo:
    def __init__(self, c=None):
        self.name: str = c.get("name") if c else ""
        self.path: str = c.get("path") if c else ""
        self.image: str = c.get("image") if c else ""
        self.meta: Dict = c.get("meta") if c else {}
        self.create_ts: int = c.get("create_ts") if c and c.get("create_ts") else 0
        self.last_access_ts: int = c.get("last_access_ts") if c and c.get("last_access_ts") else 0
        self.expiry: int = c.get("expiry") if c and c.get("expiry") else 0

    def to_str(self, indent=None):
        return json.dumps(self.__dict__, indent=indent)

    def to_json(self, indent=None):
        return json.loads(self.to_str(indent))


class Sessions(dict):
    def __init__(self, store_path: str = "", expiry: int = 3600):
        dict.__init__(self)

        store_path = store_path.strip() if store_path else ""
        store_path = store_path if store_path else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "sessions")

        self.store_path = store_path
        self.expiry = expiry if expiry > 60 else 3600
        logger.info(f"Session Path: {self.store_path}")
        logger.info(f"Session Expiry (max): {self.expiry}")

    def remove_expired(self):
        count = 0
        current_ts = int(time.time())

        if not os.path.isdir(self.store_path):
            return count

        for item in os.listdir(self.store_path):
            if os.path.isdir(os.path.join(self.store_path, item)):
                session_id = item
                session_info = self.get_session(session_id, update_ts=False, fetch_cache=False)
                expiry_ts = session_info.last_access_ts + session_info.expiry if session_info else 0

                if session_info and session_info.expiry > 0 and expiry_ts < current_ts:
                    logger.info(f"Removing expired; current ts: {current_ts}\n{session_info.to_str()}")
                    self.remove_session(session_id)
                    count += 1
                elif session_info:
                    logger.debug(
                        "Skipped {}; current ts: {}; last ts: {}; expiry: {}".format(
                            session_id, current_ts, session_info.last_access_ts, session_info.expiry
                        )
                    )
                else:
                    logger.info(f"Invalid session-id: {session_id} (will be removed)")
                    self.remove_session(session_id)
        return count

    def _session_path(self, session_id: str) -> Optional[str]:
        """Directory for ``session_id``; None unless the id is a plain directory name inside the store.

        Ids are joined onto the store path, so only letters, digits, ``_``, ``-`` and ``.`` are accepted,
        not starting with a dot (no path separators, no ``..``, no Windows drive prefix such as ``C:``).
        """
        if not isinstance(session_id, str) or not re.fullmatch(r"\w[\w.\-]*", session_id):
            return None
        return os.path.join(self.store_path, session_id)

    def get_session(self, session_id: str, update_ts: bool = True, fetch_cache: bool = True):
        path = self._session_path(session_id)
        if path is None:
            logger.warning(f"Invalid session-id: {session_id!r}")
            return None

        session_info = self.get(session_id) if fetch_cache else None
        if session_info is None:
            if os.path.exists(path):
                meta_file = os.path.join(path, "meta.info")
                if os.path.exists(meta_file):
                    with open(meta_file) as meta:
                        session_info = SessionInfo(json.loads(meta.readline()))
                    self[session_id] = session_info

        if session_info and not os.path.exists(session_info.image):
            logger.info(f"Dangling session-id: {session_id} (will be removed)")
            self.remove_session(session_id)
            session_info = None

        if session_info and update_ts:
            session_info.last_access_ts = int(time.time())
            self._write_meta_info(session_id, session_info)
        return session_info

    def remove_session(self, session_id: str):
        path = self._session_path(session_id)
        if path is None:
            logger.warning(f"Invalid session-id: {session_id!r}")
            return

        session_info = self.get(session_id)
        if session_info:
            self.pop(session_id)
        shutil.rmtree(path, ignore_errors=True)

    def add_session(self, data_file: str, expiry: int = 0, uncompress: bool = False, session_id=None):
        start = time.time()
        logger.debug(f"Load Data from: {data_file}")

        session_id = session_id if session_id else str(uuid.uuid4()).lower()
        path = self._session_path(session_id)
        if path is None:
            raise ValueError(f"Invalid session-id: {session_id!r}")

        if os.path.isdir(data_file):
            image_path = data_file
            logger.debug(f"Input Dir (Multiple Input): {image_path}")
        else:
            image_path = data_file
            logger.debug(f"Input File (Single): {image_path}")

            if uncompress:
                tmp_folder = tempfile.mkdtemp()

                logger.debug(f"UnArchive: {image_path} to {tmp_folder}")
                try:
                    safe_unpack_archive(data_file, tmp_folder)
                except Exception:
                    shutil.rmtree(tmp_folder, ignore_errors=True)
                    raise
                image_path = tmp_folder

        expiry = expiry if expiry > 0 else self.expiry

        logger.debug(f"Using Path: {path} to save session")
        os.makedirs(path, exist_ok=True)

        meta: Dict = {}
        basename = os.path.basename(image_path)

        image_file = os.path.join(path, basename)
        shutil.move(image_path, image_file)

        session_info = SessionInfo()
        session_info.name = session_id
        session_info.path = path
        session_info.image = image_file
        session_info.meta = meta
        session_info.create_ts = int(time.time())
        session_info.last_access_ts = int(time.time())
        session_info.expiry = min(expiry, self.expiry)

        self._write_meta_info(session_id, session_info)

        self[session_id] = session_info
        logger.info(f"++ Time consumed to add session {session_id}: {time.time() - start}")
        return session_id, session_info

    def _write_meta_info(self, session_id, session_info):
        path = self._session_path(session_id)
        if path is None:
            raise ValueError(f"Invalid session-id: {session_id!r}")
        meta_file = os.path.join(path, "meta.info")
        with open(meta_file, "w") as meta:
            meta.write(session_info.to_str())
