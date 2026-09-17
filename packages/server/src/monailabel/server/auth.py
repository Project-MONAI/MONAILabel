"""Password identities, revocable sessions, and project-scoped permissions."""

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import Membership, Record, Role, User
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

    def create_user(self, username: str, password: str, *, bootstrap: bool = False) -> User:
        username = username.strip().casefold()
        if not username or len(username) > 80 or len(password) < 12 or len(password) > 256:
            raise DomainError("Use a username and a password between 12 and 256 characters.")
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
        return user

    def login(self, username: str, password: str) -> tuple[User, str]:
        user = next(
            (u for u in self.store.list(User) if u.username == username.strip().casefold()), None
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
            session.insert(
                LoginSession(
                    id=hashlib.sha256(token.encode()).hexdigest(),
                    user_id=user.id,
                    expires_at=datetime.now(UTC) + timedelta(hours=12),
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
        return {
            role
            for m in self.store.list(Membership, project_id)
            if m.user_id == user.id
            for role in m.roles
        }

    def require(self, user: User, project_id: str, action: str = "read") -> None:
        roles = self.roles(user, project_id)
        allowed = {
            "read": set(Role),
            "annotate": {Role.MANAGER, Role.ANNOTATOR},
            "edit": {Role.MANAGER, Role.ANNOTATOR, Role.REVIEWER},
            "review": {Role.MANAGER, Role.REVIEWER},
            "manage": {Role.MANAGER},
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
