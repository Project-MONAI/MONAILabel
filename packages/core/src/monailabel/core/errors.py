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


class DomainError(Exception):
    """A failure that is safe to explain to a client."""

    def __init__(self, message: str, *, code: str = "invalid_request", status: int = 422):
        super().__init__(message)
        self.code = code
        self.status = status


class NotFound(DomainError):
    def __init__(self, kind: str, identifier: str):
        super().__init__(f"{kind} '{identifier}' was not found.", code="not_found", status=404)


class Conflict(DomainError):
    def __init__(self, message: str):
        super().__init__(message, code="conflict", status=409)


class Cancelled(Exception):
    """Cooperative cancellation; computed artifacts must not become visible records."""
