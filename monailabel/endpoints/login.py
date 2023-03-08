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

import logging

import requests
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from monailabel.endpoints.user.auth import Token, token_uri

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    url = token_uri()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": "monailabel-app",
        "username": form_data.username,
        "password": form_data.password,
        "grant_type": "password",
    }
    timeout = 10

    response = requests.post(url=url, headers=headers, data=data, timeout=timeout)
    response.raise_for_status()
    return response.json()
