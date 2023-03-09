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
import functools
import logging
from typing import Any, Dict, List, Union

import requests
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from monailabel.config import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str


def public_key(realm_uri) -> str:
    r = requests.get(url=realm_uri, timeout=settings.MONAI_LABEL_AUTH_TIMEOUT)
    r.raise_for_status()
    j = r.json()

    key = j["public_key"]
    return f"-----BEGIN PUBLIC KEY-----\n{key}\n-----END PUBLIC KEY-----"


@functools.cache
def open_id_configuration(realm_uri):
    response = requests.get(
        url=f"{realm_uri}/.well-known/openid-configuration",
        timeout=settings.MONAI_LABEL_AUTH_TIMEOUT,
    )
    return response.json()


def token_uri():
    return open_id_configuration(settings.MONAI_LABEL_AUTH_REALM_URI).get("token_endpoint")


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    name: Union[str, None] = None
    roles: List[str] = []


DEFAULT_USER = User(
    username="admin",
    email="admin@monailabel.com",
    name="UNK",
    roles=[
        settings.MONAI_LABEL_AUTH_ROLE_ADMIN,
        settings.MONAI_LABEL_AUTH_ROLE_RESEARCHER,
        settings.MONAI_LABEL_AUTH_ROLE_REVIEWER,
        settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR,
        settings.MONAI_LABEL_AUTH_ROLE_USER,
    ],
)


def from_token(token: str):
    if not settings.MONAI_LABEL_AUTH_ENABLE:
        return DEFAULT_USER

    options = {
        "verify_signature": True,
        "verify_aud": False,
        "verify_exp": True,
    }

    key = public_key(settings.MONAI_LABEL_AUTH_REALM_URI)
    payload = jwt.decode(token, key, options=options)

    username: str = payload.get(settings.MONAI_LABEL_AUTH_TOKEN_USERNAME)
    email: str = payload.get(settings.MONAI_LABEL_AUTH_TOKEN_EMAIL)
    name: str = payload.get(settings.MONAI_LABEL_AUTH_TOKEN_NAME)
    realm_access: Dict[str, Any] = payload.get(settings.MONAI_LABEL_AUTH_TOKEN_REALM_ACCESS)
    roles = [] if not realm_access else realm_access.get(settings.MONAI_LABEL_AUTH_TOKEN_ROLES)

    return User(username=username, email=email, name=name, roles=roles)


async def get_current_user(token: str = Depends(oauth2_scheme) if settings.MONAI_LABEL_AUTH_ENABLE else ""):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        return from_token(token)
    except JWTError as e:
        logger.error(e)
        raise credentials_exception


def _validate_role(user, role):
    if not settings.MONAI_LABEL_AUTH_ENABLE:
        return user

    if role not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Role "{role}" is required to perform this action',
        )
    return user


async def get_admin_user(user: User = Security(get_current_user)):
    return _validate_role(user, settings.MONAI_LABEL_AUTH_ROLE_ADMIN)


async def get_researcher_user(user: User = Security(get_current_user)):
    return _validate_role(user, settings.MONAI_LABEL_AUTH_ROLE_RESEARCHER)


async def get_reviwer_user(user: User = Security(get_current_user)):
    return _validate_role(user, settings.MONAI_LABEL_AUTH_ROLE_REVIEWER)


async def get_annotator_user(user: User = Security(get_current_user)):
    return _validate_role(user, settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR)


async def get_basic_user(user: User = Security(get_current_user)):
    return _validate_role(user, settings.MONAI_LABEL_AUTH_ROLE_USER)
