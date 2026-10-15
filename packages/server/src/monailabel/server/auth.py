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

"""Password identities, revocable sessions, and project-scoped permissions."""

import hashlib
import hmac
import re
import secrets
from collections import deque
from datetime import UTC, datetime, timedelta
from threading import Lock
from time import monotonic

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import Membership, Record, Role, User
from monailabel.server.demo_access import (
    DemoAccess,
    DemoBusy,
    access,
    current_round,
    demo_minutes,
    public_registration_enabled,
    registration_enabled,
    require_active,
    require_space,
    visit,
)
from monailabel.server.demo_registration import create_project
from monailabel.server.login_throttle import LoginThrottle
from monailabel.server.metadata_quota import MetadataQuota
from monailabel.server.storage import Store


class Password(Record):
    salt: str
    digest: str


class LoginSession(Record):
    user_id: str
    expires_at: datetime


def digest(password: str, salt: str) -> str:
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


class Auth:
    def __init__(self, store: Store):
        self.store = store
        self.registration_lock = Lock()
        self.registration_attempts: deque[float] = deque(maxlen=10)
        self.login_throttle = LoginThrottle()

    def register_demo(self, username: str, password: str) -> tuple[User, str | None]:
        if not registration_enabled():
            raise DomainError("Self-registration is not enabled on this server.", status=403)
        # Bound unauthenticated password hashing; no client-controlled IP headers.
        if not self.registration_lock.acquire(blocking=False):
            raise DemoBusy("Another registration is being processed. Please retry shortly.", 5)
        try:
            now_tick = monotonic()
            if (
                len(self.registration_attempts) == 10
                and now_tick - self.registration_attempts[0] < 60
            ):
                raise DemoBusy("Too many registration attempts. Please retry in one minute.", 60)
            self.registration_attempts.append(now_tick)
            username = username.strip().casefold()
            if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,39}", username):
                raise DomainError(
                    "Use 3–40 letters, numbers, dots, underscores or hyphens for your username."
                )
            if not 12 <= len(password) <= 256:
                raise DomainError("Use a password between 12 and 256 characters.")
            salt = secrets.token_hex(16)
            hashed = digest(password, salt)
            token = secrets.token_urlsafe(32)
            with self.store.transaction() as session:
                users = session.list(User)
                if not any(user.is_admin for user in users):
                    raise DomainError(
                        "The host must finish setting up the server first.", status=503
                    )
                if len(session.list(DemoAccess)) >= 1000:
                    raise DomainError(
                        "The host must clean up previous demo visits "
                        "before more accounts can be created.",
                        status=503,
                    )
                if any(user.username == username for user in users):
                    raise Conflict(
                        "That username is already used. Sign in, or choose a new username."
                    )
                automatic = public_registration_enabled()
                if not automatic:
                    require_space(session)
                now = datetime.now(UTC)
                user = User(username=username)
                minutes = demo_minutes()
                deadline = now + timedelta(minutes=minutes)
                session.insert(user)
                session.insert(Password(id=user.id, salt=salt, digest=hashed))
                project = None if automatic else create_project(session, user.id)
                lease = DemoAccess(
                    id=user.id,
                    minutes=minutes,
                    started_at=None if automatic else now,
                    expires_at=None if automatic else deadline,
                    self_registered=True,
                    initial_project_id=project.id if project else None,
                    round_id=current_round(session).id,
                    restricted_roles=automatic,
                )
                session.insert(lease)
                visit(session, user, lease)
                if not automatic:
                    session.insert(
                        LoginSession(
                            id=hashlib.sha256(token.encode()).hexdigest(),
                            user_id=user.id,
                            expires_at=deadline,
                        )
                    )
                MetadataQuota(session.connection).backfill(user.id)
            return user, None if automatic else token
        finally:
            self.registration_lock.release()

    def create_user(
        self,
        username: str,
        password: str,
        *,
        bootstrap: bool = False,
        demo_minutes: int | None = None,
    ) -> User:
        if demo_minutes is not None and (bootstrap or demo_minutes not in {60, 120}):
            raise DomainError("Demo visitors must have a 60 or 120 minute invitation.")
        username = username.strip().casefold()
        if not username or len(username) > 80 or len(password) < 12 or len(password) > 256:
            raise DomainError("Use a username and a password between 12 and 256 characters.")
        # The public setup endpoint must not remain an unthrottled hash oracle
        # after bootstrap. The transaction below still handles concurrent setup.
        if bootstrap and self.store.list(User):
            raise Conflict("An administrator already exists. Sign in instead.")
        salt = secrets.token_hex(16)
        hashed = digest(password, salt)
        with self.store.transaction() as session:
            users = session.list(User)
            if bootstrap and users:
                raise Conflict("An administrator already exists. Sign in instead.")
            if any(u.username == username for u in users):
                raise Conflict("Username already exists.")
            user = User(username=username, is_admin=bootstrap)
            session.insert(user)
            session.insert(Password(id=user.id, salt=salt, digest=hashed))
            if demo_minutes is not None:
                lease = DemoAccess(
                    id=user.id, minutes=demo_minutes, round_id=current_round(session).id
                )
                session.insert(lease)
                visit(session, user, lease)
                MetadataQuota(session.connection).backfill(user.id)
        return user

    def login(
        self, username: str, password: str, *, client_address: str = "local"
    ) -> tuple[User, str]:
        if not 1 <= len(username) <= 80 or not 1 <= len(password) <= 256:
            raise DomainError("Invalid username or password.", status=401)
        with self.login_throttle.attempt(username, client_address):
            user = next(
                (u for u in self.store.list(User) if u.username == username.strip().casefold()),
                None,
            )
            stored = self.store.get(Password, user.id) if user else None
            hashed = digest(password, stored.salt if stored else "00" * 16)
            if (
                not user
                or not user.active
                or not stored
                or not hmac.compare_digest(hashed, stored.digest)
            ):
                raise DomainError("Invalid username or password.", status=401)
            return user, self.issue(user)

    def issue(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        with self.store.transaction() as session:
            user = session.get(User, user.id)
            if not user.active:
                raise DomainError("This account is disabled.", status=401)
            lease = require_active(session, user)
            now = datetime.now(UTC)
            expires_at = now + timedelta(hours=12)
            if lease:
                if lease.expires_at is None:
                    # Invitation admission is independent of whether anonymous
                    # visitors are allowed to create their own accounts.
                    require_space(session)
                    if lease.restricted_roles and lease.initial_project_id is None:
                        project = create_project(session, user.id, restricted_roles=True)
                        lease = lease.model_copy(update={"initial_project_id": project.id})
                    lease = lease.model_copy(
                        update={
                            "started_at": now,
                            "expires_at": now + timedelta(minutes=lease.minutes),
                        }
                    )
                    session.update(lease)
                    visit(session, user, lease)
                    MetadataQuota(session.connection).backfill(user.id)
                assert lease.expires_at is not None
                expires_at = min(expires_at, lease.expires_at)
            session.insert(
                LoginSession(
                    id=hashlib.sha256(token.encode()).hexdigest(),
                    user_id=user.id,
                    expires_at=expires_at,
                )
            )
        return token

    def authenticate(self, token: str | None) -> User:
        if not token:
            raise DomainError("Sign in to continue.", status=401)
        try:
            session = self.store.get(LoginSession, hashlib.sha256(token.encode()).hexdigest())
            user = self.store.get(User, session.user_id)
        except DomainError as exc:
            raise DomainError("Session is invalid. Sign in again.", status=401) from exc
        require_active(self.store, user)
        if session.expires_at <= datetime.now(UTC) or not user.active:
            raise DomainError("Session expired or user disabled. Sign in again.", status=401)
        return user

    def revoke(self, token: str) -> None:
        with self.store.transaction() as session:
            item = session.get(LoginSession, hashlib.sha256(token.encode()).hexdigest())
            session.update(item.model_copy(update={"expires_at": datetime.now(UTC)}))

    def roles(self, user: User, project_id: str) -> set[Role]:
        if user.is_admin:
            return set(Role)
        roles = {
            role
            for m in self.store.list(Membership, project_id)
            if m.user_id == user.id
            for role in m.roles
        }
        lease = access(self.store, user.id)
        if lease and lease.restricted_roles:
            return roles & {Role.ANNOTATOR, Role.REVIEWER}
        return roles

    def require(self, user: User, project_id: str, action: str = "read") -> None:
        from monailabel.server.demo_policy import setup_allowed

        require_active(self.store, user)
        roles = self.roles(user, project_id)
        allowed = {
            "read": set(Role),
            "annotate": {Role.MANAGER, Role.ANNOTATOR},
            "edit": {Role.MANAGER, Role.ANNOTATOR, Role.REVIEWER},
            "review": {Role.MANAGER, Role.REVIEWER},
            "manage": {Role.MANAGER},
            "create_setup": (
                {Role.MANAGER, Role.ANNOTATOR}
                if action == "create_setup" and setup_allowed(self.store, project_id, user.id)
                else {Role.MANAGER}
            ),
        }[action]
        if not roles & allowed:
            raise DomainError("Your project role does not allow this action.", status=403)

    def add_member(self, project_id: str, user_id: str, roles: list[Role]) -> Membership:
        self.store.get(User, user_id)
        with self.store.transaction() as session:
            previous = next(
                (m for m in session.list(Membership, project_id) if m.user_id == user_id), None
            )
            if previous:
                member = previous.model_copy(update={"roles": roles})
                session.update(member)
            else:
                member = Membership(project_id=project_id, user_id=user_id, roles=roles)
                session.insert(member)
        return member
