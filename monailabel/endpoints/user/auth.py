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
from typing import Any, Dict, List, Union

import requests
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

REALM_URI = "http://localhost:8080/realms/monailabel"
TIMEOUT = 5
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str


def public_key(realm_uri=REALM_URI, timeout=TIMEOUT) -> str:
    r = requests.get(url=realm_uri, timeout=timeout)
    r.raise_for_status()
    j = r.json()

    print(json.dumps(j, indent=2))
    key = j["public_key"]
    return f"-----BEGIN PUBLIC KEY-----\n{key}\n-----END PUBLIC KEY-----"


def open_id_configuration() -> dict:
    response = requests.get(url=f"{REALM_URI}/.well-known/openid-configuration", timeout=TIMEOUT)
    j = response.json()
    print(json.dumps(j, indent=2))
    return j


def token_uri():
    return open_id_configuration().get("token_endpoint")


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    name: Union[str, None] = None
    roles: List[str] = []


def from_token(token: str):
    options = {
        "verify_signature": True,
        "verify_aud": False,
        "verify_exp": True,
    }

    key = public_key()
    payload = jwt.decode(token, key, options=options)

    username: str = payload.get("preferred_username")
    email: str = payload.get("email")
    name: str = payload.get("name")
    realm_access: Dict[str, Any] = payload.get("realm_access")
    roles = [] if not realm_access else realm_access.get("roles")

    return User(username=username, email=email, name=name, roles=roles)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        return from_token(token)
    except JWTError as e:
        print(e)
        raise credentials_exception


def _validate_role(user, role):
    if role not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Role "{role}" is required to perform this action',
        )
    return user


async def get_admin_user(user: User = Security(get_current_user)):
    return _validate_role(user, "monailabel-admin")


async def get_reviwer_user(user: User = Security(get_current_user)):
    return _validate_role(user, "monailabel-reviewer")


async def get_annotator_user(user: User = Security(get_current_user)):
    return _validate_role(user, "monailabel-annotator")


async def get_basic_user(user: User = Security(get_current_user)):
    return _validate_role(user, "monailabel-user")
