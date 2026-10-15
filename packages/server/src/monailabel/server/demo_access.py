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

"""Opt-in, absolute visitor deadlines; independent of browser/session lifetime."""

import math
import os
from datetime import UTC, datetime

from monailabel.core.errors import DomainError, NotFound
from monailabel.core.models import TERMINAL_STATUSES, Job, Membership, Record, User
from monailabel.server.desktops.models import DesktopSession
from monailabel.server.storage import Session, Store


class DemoAccess(Record):
    minutes: int
    started_at: datetime | None = None
    expires_at: datetime | None = None
    self_registered: bool = False
    initial_project_id: str | None = None
    round_id: str | None = None
    stopped_at: datetime | None = None
    restricted_roles: bool = False


class DemoRound(Record):
    pass


class DemoVisit(Record):
    """Durable feedback identity, not an account or a login credential."""

    username: str
    round_id: str
    started_at: datetime | None = None
    expires_at: datetime | None = None
    reset_at: datetime | None = None


def current_round(session: Session) -> DemoRound:
    rounds = session.list(DemoRound)
    if rounds:
        return rounds[-1]
    round_ = DemoRound()
    session.insert(round_)
    return round_


def visit(session: Session, user: User, lease: DemoAccess) -> None:
    identity = DemoVisit(
        id=user.id,
        created_at=user.created_at,
        username=user.username,
        round_id=lease.round_id or current_round(session).id,
        started_at=lease.started_at,
        expires_at=lease.expires_at,
    )
    if any(item.id == user.id for item in session.list(DemoVisit)):
        session.update(identity)
    else:
        session.insert(identity)


def lifecycle_enabled() -> bool:
    return os.environ.get("MONAILABEL_DEMO_LIFECYCLE") == "1"


def demo_minutes() -> int:
    value = os.environ.get("MONAILABEL_DEMO_MINUTES", "60")
    if value not in {"60", "120"}:
        raise DomainError("The host must choose a 60 or 120 minute demo.", status=503)
    return int(value)


class DemoBusy(DomainError):
    def __init__(self, message: str, retry_after: int, *, code: str = "demo_busy"):
        super().__init__(message, code=code, status=429)
        self.retry_after = retry_after


def registration_enabled() -> bool:
    return all(
        os.environ.get(name) == "1"
        for name in (
            "MONAILABEL_DEMO_SELF_REGISTER",
            "MONAILABEL_DEMO_CACHE_ONLY",
            "MONAILABEL_DEMO_TRAINING_PREVIEW",
        )
    )


def public_registration_enabled() -> bool:
    return registration_enabled() and all(
        os.environ.get(name) == "1"
        for name in (
            "MONAILABEL_DEMO_PUBLIC_REGISTER",
            "MONAILABEL_DEMO_HTTP_LIMITS",
            "MONAILABEL_DEMO_LIFECYCLE",
        )
    )


def capacity() -> int:
    try:
        value = int(os.environ.get("MONAILABEL_DEMO_CAPACITY", "2"))
    except ValueError:
        value = 0
    if not 1 <= value <= 8:
        raise DomainError("The host must configure the demo capacity.", status=503)
    return value


def admission(store: Store | Session) -> dict[str, int | bool]:
    now = datetime.now(UTC)
    leases = store.list(DemoAccess)
    active = {lease.id for lease in leases if lease.expires_at and lease.expires_at > now}
    if lifecycle_enabled():
        active.update(lease.id for lease in leases if lease.started_at and lease.stopped_at is None)
    # Expired accounts with unfinished resources still occupy a slot. Access expiry
    # alone must not be mistaken for GPU/process cleanup.
    busy_projects = {
        job.project_id for job in store.list(Job) if job.status not in TERMINAL_STATUSES
    }
    busy_users = {m.user_id for m in store.list(Membership) if m.project_id in busy_projects}
    busy_users.update(d.user_id for d in store.list(DesktopSession) if not d.ended)
    active.update(lease.id for lease in leases if lease.id in busy_users)
    upcoming = [
        math.ceil((lease.expires_at - now).total_seconds())
        for lease in leases
        if lease.expires_at and lease.expires_at > now
    ]
    limit = capacity()
    return {
        "capacity": limit,
        "active": len(active),
        "busy": len(active) >= limit,
        "retry_after_seconds": max(1, min(upcoming, default=60)),
        "minutes": demo_minutes(),
    }


def require_space(store: Store | Session) -> None:
    state = admission(store)
    if state["busy"]:
        retry = int(state["retry_after_seconds"])
        raise DemoBusy(
            f"The demo is busy with {state['active']} active visits or unfinished sessions. "
            f"Please try again in about {math.ceil(retry / 60)} minutes. No new visit was started.",
            retry,
        )


def access(store: Store | Session, user_id: str) -> DemoAccess | None:
    try:
        return store.get(DemoAccess, user_id)
    except NotFound:
        return None


def require_active(store: Store | Session, user: User) -> DemoAccess | None:
    lease = access(store, user.id)
    if lease and lease.expires_at and lease.expires_at <= datetime.now(UTC):
        raise DomainError(
            "This demo account has expired. Create a new demo account when a place is available."
            if lease.self_registered
            else "This demo account has expired. Ask the host for a new invitation.",
            code="demo_expired",
            status=401,
        )
    return lease
