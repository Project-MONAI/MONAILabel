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

from monailabel.config import settings
from monailabel.endpoints.user.auth import Token, User, get_current_user, token_uri

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
    responses={404: {"description": "Not found"}},
)

# https://auth0.com/
# https://www.keycloak.org/


@router.get("/", summary="Check If Auth is Enabled")
async def auth_enabled():
    return {
        "enabled": settings.MONAI_LABEL_AUTH_ENABLE,
        "client_id": settings.MONAI_LABEL_AUTH_CLIENT_ID,
        "realm": settings.MONAI_LABEL_AUTH_REALM_URI,
    }


@router.post("/token", response_model=Token, summary="Fetch new access code/token")
async def access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if not settings.MONAI_LABEL_AUTH_ENABLE:
        return {"access_token": None, "token_type": None}

    url = token_uri()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": settings.MONAI_LABEL_AUTH_CLIENT_ID,
        "username": form_data.username,
        "password": form_data.password,
        "grant_type": "password",
    }
    timeout = 30

    response = requests.post(url=url, headers=headers, data=data, timeout=timeout)
    response.raise_for_status()
    return response.json()


@router.get("/token/valid", summary="Check If current token is Valid")
async def valid_token(user: User = Depends(get_current_user)):
    return user.dict()
