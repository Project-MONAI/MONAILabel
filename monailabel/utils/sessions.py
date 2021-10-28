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

import json
import logging
import os
import pathlib
import shutil
import tempfile
import time
import uuid
from typing import Dict

logger = logging.getLogger(__name__)


class SessionInfo(object):
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
        for item in os.listdir(self.store_path):
            if os.path.isdir(os.path.join(self.store_path, item)):
                session_id = item
                session_info = self.get_session(session_id, update_ts=False, fetch_cache=False)
                expiry_ts = session_info.last_access_ts + session_info.expiry

                if session_info and session_info.expiry > 0 and expiry_ts < current_ts:
                    logger.info("Removing expired; current ts: {}\n{}".format(current_ts, session_info.to_str()))
                    self.remove_session(session_id)
                    count += 1
                elif session_info:
                    logger.debug(
                        "Skipped {}; current ts: {}; last ts: {}; expiry: {}".format(
                            session_id, current_ts, session_info.last_access_ts, session_info.expiry
                        )
                    )
                else:
                    logger.info("Invalid session-id: {} (will be removed)".format(session_id))
                    self.remove_session(session_id)
        return count

    def get_session(self, session_id: str, update_ts: bool = True, fetch_cache: bool = True):
        session_info = self.get(session_id) if fetch_cache else None
        if session_info is None:
            path = os.path.join(self.store_path, session_id)
            if os.path.exists(path):
                meta_file = os.path.join(path, "meta.info")
                if os.path.exists(meta_file):
                    with open(meta_file, "r") as meta:
                        session_info = SessionInfo(json.loads(meta.readline()))
                    self[session_id] = session_info

        if session_info and not os.path.exists(session_info.image):
            logger.info("Dangling session-id: {} (will be removed)".format(session_id))
            self.remove_session(session_id)
            session_info = None

        if session_info and update_ts:
            session_info.last_access_ts = int(time.time())
            self._write_meta_info(session_id, session_info)
        return session_info

    def remove_session(self, session_id: str):
        session_info = self.get(session_id)
        if session_info:
            self.pop(session_id)
        path = os.path.join(self.store_path, session_id)
        shutil.rmtree(path, ignore_errors=True)

    def add_session(self, data_file: str, expiry: int = 0, uncompress: bool = False):
        start = time.time()
        logger.debug("Load Data from: {}".format(data_file))

        if os.path.isdir(data_file):
            image_path = data_file
            logger.debug(f"Input Dir (Multiple Input): {image_path}")
        else:
            image_path = data_file
            logger.debug(f"Input File (Single): {image_path}")

            if uncompress:
                tmp_folder = tempfile.TemporaryDirectory().name
                os.makedirs(tmp_folder, exist_ok=True)

                logger.debug(f"UnArchive: {image_path} to {tmp_folder}")
                shutil.unpack_archive(data_file, tmp_folder)
                image_path = tmp_folder

        session_id = str(uuid.uuid1()).lower()
        path = os.path.join(self.store_path, session_id)
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
        path = os.path.join(self.store_path, session_id)
        meta_file = os.path.join(path, "meta.info")
        with open(meta_file, "w") as meta:
            meta.write(session_info.to_str())
