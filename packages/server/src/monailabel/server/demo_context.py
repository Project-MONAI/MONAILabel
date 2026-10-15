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

"""Request/job identity, propagated through AnyIO's copied thread context."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from monailabel.core.preparation import prepared_models

actor: ContextVar[str | None] = ContextVar("monailabel_actor", default=None)


@contextmanager
def acting_as(user_id: str | None, *, prepared_only: bool = False) -> Iterator[None]:
    token = actor.set(user_id)
    try:
        with prepared_models(prepared_only):
            yield
    finally:
        actor.reset(token)
