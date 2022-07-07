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

from fastapi import APIRouter, Depends

from monailabel.endpoints.user.auth import User, get_basic_user
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance

router = APIRouter(
    prefix="/info",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


def app_info():
    instance: MONAILabelApp = app_instance()
    return instance.info()


@router.get("/", summary="Get App Info")
async def api_app_info(user: User = Depends(get_basic_user)):
    return app_info()
