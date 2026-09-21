"""Account-owned browser desktops with reconnectable, private native drafts."""

import hashlib
import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from monailabel.core.errors import DomainError
from monailabel.core.models import Asset, User
from monailabel.server.auth import Auth, LoginSession
from monailabel.server.desktops.models import DesktopSession
from monailabel.server.jobs import JobContext
from monailabel.server.storage import Store
from monailabel.viewers.browser_desktop import BrowserDesktop

logger = logging.getLogger(__name__)
CLOSE_DELAY_SECONDS = 10.0


class DesktopSessions:
    def __init__(self, store: Store, auth: Auth, directory: Path):
        self.store = store
        self.auth = auth
        namespace = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
        self.runtime = BrowserDesktop(directory / "viewers" / "desktop", namespace)
        # Serialize provisioning and slot allocation without blocking connected viewers.
        self.launch_lock = threading.Lock()
        self.lock = threading.RLock()
        self.starting: set[str] = set()
        self.connections: dict[str, int] = {}
        self.pending_close: dict[str, threading.Timer] = {}
        self.closing = False

    def connected(self, identifier: str, user: User) -> None:
        with self.lock:
            self.owned(identifier, user)
            self.connections[identifier] = self.connections.get(identifier, 0) + 1
            self.cancel_close(identifier)

    def disconnected(self, identifier: str) -> None:
        with self.lock:
            remaining = self.connections.get(identifier, 1) - 1
            if remaining:
                self.connections[identifier] = remaining
            else:
                self.connections.pop(identifier, None)

    def close_tab(self, identifier: str, user: User) -> None:
        with self.lock:
            desktop = self.store.get(DesktopSession, identifier)
            if desktop.user_id != user.id:
                raise DomainError("This desktop belongs to another account.", status=403)
            if self.closing or desktop.ended:
                return
            self.cancel_close(identifier)
            timer = threading.Timer(CLOSE_DELAY_SECONDS, self.end_disconnected, args=(identifier,))
            timer.daemon = True
            self.pending_close[identifier] = timer
            timer.start()

    def cancel_close(self, identifier: str) -> None:
        if timer := self.pending_close.pop(identifier, None):
            timer.cancel()

    def end_disconnected(self, identifier: str) -> None:
        with self.lock:
            # A refresh reconnects during the grace period. Another open tab also
            # keeps the viewer alive; a network disconnect alone never ends a draft.
            if self.closing or self.pending_close.get(identifier) is not threading.current_thread():
                return
            self.pending_close.pop(identifier)
            if self.connections.get(identifier):
                return
            try:
                desktop = self.store.get(DesktopSession, identifier)
                if not desktop.ended:
                    self._end(desktop)
            except Exception:
                logger.exception("Could not end browser desktop after its tab closed")

    def close(self) -> None:
        with self.lock:
            self.closing = True
            for timer in self.pending_close.values():
                timer.cancel()
            self.pending_close.clear()

    def owned(self, identifier: str, user: User) -> DesktopSession:
        if not user.active:
            raise DomainError("Your account is disabled.", status=403)
        desktop = self.store.get(DesktopSession, identifier)
        if desktop.user_id != user.id:
            raise DomainError("This desktop belongs to another account.", status=403)
        if desktop.ended:
            raise DomainError("This desktop session has ended. Open the sample again.", status=410)
        self.auth.require(
            user, desktop.project_id, "review" if desktop.mode == "review" else "read"
        )
        self.store.get(Asset, desktop.asset_id)
        return desktop

    def inspect(self, identifier: str, user: User) -> DesktopSession:
        desktop = self.owned(identifier, user)
        if identifier in self.starting:
            raise DomainError("The desktop is still starting. Wait for its launch job.", status=409)
        if not self.runtime.running(identifier):
            self.end(identifier, user)
            raise DomainError("The desktop has stopped. Open the sample again.", status=410)
        return desktop

    def renew(self, desktop: DesktopSession) -> None:
        # The native bridge keeps its token in memory. Reconnecting an authenticated
        # browser renews it without restarting the viewer or replacing its draft.
        with self.store.transaction() as session:
            if session.get(DesktopSession, desktop.id).ended:
                raise DomainError("The desktop session has ended.", status=410)
            credential = session.get(LoginSession, desktop.credential_id)
            session.update(
                credential.model_copy(
                    update={"expires_at": datetime.now(UTC) + timedelta(hours=12)}
                )
            )

    def open(
        self,
        asset: Asset,
        user: User,
        viewer: Literal["slicer", "qupath"],
        mode: Literal["annotation", "review"],
        backend_url: str,
        context: JobContext,
    ) -> DesktopSession:
        with self.launch_lock:
            # Recheck after any provisioning queue delay or account/role change.
            context.progress(0, "Preparing the browser desktop.")
            user = self.store.get(User, user.id)
            if not user.active:
                raise DomainError("Your account is disabled.", status=403)
            self.auth.require(user, asset.project_id, "review" if mode == "review" else "read")
            asset = self.store.get(Asset, asset.id)
            active = [s for s in self.store.list(DesktopSession) if not s.ended]
            for previous in active:
                if not self.runtime.running(previous.id):
                    with self.lock:
                        self._end(previous)
                    continue
                if (previous.user_id, previous.asset_id, previous.viewer, previous.mode) == (
                    user.id,
                    asset.id,
                    viewer,
                    mode,
                ):
                    with self.lock:
                        self.renew(previous)
                        return previous
            active = [s for s in self.store.list(DesktopSession) if not s.ended]
            limit = max(1, int(os.environ.get("MONAILABEL_DESKTOP_LIMIT", "8")))
            if len(active) >= limit:
                raise DomainError(
                    "All desktop slots are in use. End an unused session from Browser desktops.",
                    status=409,
                )
            token = self.auth.issue(user)
            desktop = DesktopSession(
                project_id=asset.project_id,
                user_id=user.id,
                asset_id=asset.id,
                viewer=viewer,
                mode=mode,
                credential_id=hashlib.sha256(token.encode()).hexdigest(),
            )
            self.starting.add(desktop.id)
            try:
                with self.store.transaction() as session:
                    session.insert(desktop)

                def progress(message: str) -> None:
                    context.progress(0.5, message)
                    self.owned(desktop.id, self.store.get(User, user.id))

                self.runtime.start(
                    desktop.id,
                    viewer,
                    {
                        "url": backend_url,
                        "project_id": asset.project_id,
                        "asset_id": asset.id,
                        "mode": mode,
                        "shared_filesystem": False,
                        "token": token,
                    },
                    progress,
                )
                with self.lock:
                    user = self.store.get(User, user.id)
                    self.owned(desktop.id, user)
                    context.progress(1, "Browser desktop is ready.")
                    return desktop
            except Exception:
                try:
                    # An explicit end during setup can precede container creation.
                    with self.lock:
                        self._end(desktop)
                except Exception:
                    logger.exception("Could not clean up a failed browser desktop launch")
                raise
            finally:
                self.starting.discard(desktop.id)

    def end(self, identifier: str, user: User) -> None:
        with self.lock:
            desktop = self.store.get(DesktopSession, identifier)
            if desktop.user_id != user.id:
                raise DomainError("This desktop belongs to another account.", status=403)
            if not desktop.ended:
                self._end(desktop)

    def _end(self, desktop: DesktopSession) -> None:
        self.cancel_close(desktop.id)
        # Revoke even when Docker is unavailable; keep a failed stop retryable.
        with self.store.transaction() as session:
            credential = session.get(LoginSession, desktop.credential_id)
            session.update(credential.model_copy(update={"expires_at": datetime.now(UTC)}))
        self.runtime.stop(desktop.id)
        with self.store.transaction() as session:
            session.update(desktop.model_copy(update={"ended": True}))
