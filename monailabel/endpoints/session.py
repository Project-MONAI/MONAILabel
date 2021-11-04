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
import logging
import os
import shutil
import tempfile
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.others.generic import get_basename, get_mime_type, remove_file
from monailabel.utils.sessions import Sessions

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/session",
    tags=["Session"],
    responses={404: {"description": "Not found"}},
)


def get_session(session_id: str, update_ts: bool = False, image: bool = False):
    instance: MONAILabelApp = app_instance()
    sessions: Sessions = instance.sessions()
    if sessions is None:
        logger.error("Session Feature is Not Enabled")
        raise HTTPException(status_code=406, detail="Session Feature is Not Enabled")

    session_info = sessions.get_session(session_id, update_ts=update_ts)
    if session_info:
        if image:
            return FileResponse(
                session_info.image,
                media_type=get_mime_type(session_info.image),
                filename=get_basename(session_info.image),
            )
        return session_info.to_json()
    raise HTTPException(status_code=404, detail=f"Session ({session_id}) Not Found")


def create_session(
    background_tasks: BackgroundTasks,
    uncompress: bool = False,
    expiry: int = 0,
    files: List[UploadFile] = File(...),
):
    instance: MONAILabelApp = app_instance()
    sessions: Sessions = instance.sessions()
    if sessions is None:
        logger.error("Session Feature is Not Enabled")
        raise HTTPException(status_code=406, detail="Session Feature is Not Enabled")

    logger.info(f"Uncompress: {uncompress}; Expiry: {expiry}")
    logger.info(f"Request Files: {files}")

    received_dir = tempfile.NamedTemporaryFile().name
    os.makedirs(received_dir, exist_ok=True)

    input_image = ""
    total_files = 0
    for f in files:
        basename = get_basename(f.filename) if f.filename else tempfile.NamedTemporaryFile().name
        input_image = os.path.join(received_dir, basename)
        with open(input_image, "wb") as fb:
            shutil.copyfileobj(f.file, fb)

        total_files += 1
        logger.info(f"{total_files} => {f} => {input_image}")

    if total_files > 1:
        logger.info(f"Input has multiple files; Saving ALL into: {received_dir}")
        input_image = received_dir

    session_id, session_info = sessions.add_session(input_image, expiry, uncompress)
    background_tasks.add_task(remove_file, received_dir)

    if total_files == 0:
        raise HTTPException(status_code=404, detail="Image(s) Not Found")

    logger.info(f"Session ID: {session_id}; Info: {session_info.to_str()}")
    return {"session_id": session_id, "session_info": session_info.to_json()}


def remove_session(session_id: str):
    instance: MONAILabelApp = app_instance()
    sessions: Sessions = instance.sessions()
    if sessions is None:
        logger.error("Session Feature is Not Enabled")
        raise HTTPException(status_code=406, detail="Session Feature is Not Enabled")

    session_info = sessions.get_session(session_id)
    if session_info:
        sessions.remove_session(session_id)
        return session_info.to_json()
    raise HTTPException(status_code=404, detail="Session Not Found")


@router.get("/{session_id}", summary="Get Session ID")
async def api_get_session(session_id: str, update_ts: bool = False, image: bool = False):
    return get_session(session_id, update_ts, image)


@router.put("/", summary="Create new session with Image")
async def api_create_session(
    background_tasks: BackgroundTasks,
    uncompress: bool = False,
    expiry: int = 0,
    files: List[UploadFile] = File(...),
):
    return create_session(background_tasks, uncompress, expiry, files)


@router.delete("/{session_id}", summary="Delete Session")
async def api_remove_session(session_id: str):
    return remove_session(session_id)
