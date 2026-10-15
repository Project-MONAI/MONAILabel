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

"""Timed-visit cleanup and an explicit, confirmed, idle-only demo reset."""

import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import TERMINAL_STATUSES, Job, Membership, Project, Record, User
from monailabel.server.auth import LoginSession
from monailabel.server.deletion import Deletion
from monailabel.server.demo_access import DemoAccess, DemoRound, DemoVisit, current_round, visit
from monailabel.server.desktops.models import DesktopSession
from monailabel.server.storage import Session
from monailabel.server.video.models import VideoEditor

if TYPE_CHECKING:
    from monailabel.server.service import Services

logger = logging.getLogger(__name__)


class DemoFeedback(Record):
    round_id: str
    source_kind: str
    source_id: str
    content: dict[str, JsonValue]


class DemoLifecycle:
    def __init__(self, services: "Services", *, enabled: bool):
        self.services = services
        self.enabled = enabled
        self.stopping = threading.Event()
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        if enabled:
            with services.store.transaction() as session:
                for lease in session.list(DemoAccess):
                    user = session.get(User, lease.id)
                    if not user.is_admin:
                        visit(session, user, lease)
            self.thread = threading.Thread(target=self._run, name="demo-expiry", daemon=True)
            self.thread.start()

    def close(self) -> None:
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=30)

    def _run(self) -> None:
        while not self.stopping.wait(0.5):
            try:
                self.sweep()
            except Exception:
                logger.exception("Demo expiry cleanup failed; access remains expired; will retry")

    @staticmethod
    def private_projects(session: Session, users: set[str]) -> set[str]:
        memberships = session.list(Membership)
        result = set()
        for project in session.list(Project):
            members = {m.user_id for m in memberships if m.project_id == project.id}
            if members and members <= users:
                result.add(project.id)
        return result

    def sweep(self) -> None:
        """Authentication rejects precisely at expiry; this reaps resources without a browser."""
        if not self.enabled:
            return
        service = self.services
        with self.lock:
            now = datetime.now(UTC)
            with service.store.transaction() as session:
                users = {u.id: u for u in session.list(User) if not u.is_admin}
                expired = {
                    d.id
                    for d in session.list(DemoAccess)
                    if d.id in users
                    and d.expires_at
                    and d.expires_at <= now
                    and d.stopped_at is None
                }
                projects = self.private_projects(session, expired)
            if not expired:
                return
            for job in service.store.list(Job):
                if (
                    job.user_id in expired or (job.user_id is None and job.project_id in projects)
                ) and job.status not in TERMINAL_STATUSES:
                    service.jobs.cancel(job.id)
            for desktop in service.store.list(DesktopSession):
                if desktop.user_id in expired and not desktop.ended:
                    try:
                        service.desktops.end(desktop.id, users[desktop.user_id])
                    except Exception:
                        logger.exception(
                            "Timed desktop stop failed; retaining its capacity reservation"
                        )
            # Revoke CVAT access immediately, but do not pretend a shared-service
            # import has stopped merely because its MONAI job was cancelled.
            for editor in service.store.list(VideoEditor):
                if not (
                    editor.user_id in expired
                    or (editor.user_id is None and editor.project_id in projects)
                ):
                    continue
                if not editor.ended:
                    with service.store.transaction() as session:
                        current = session.get(VideoEditor, editor.id)
                        session.update(current.model_copy(update={"ended": True, "ready": False}))
                if editor.preparation_request_id:
                    try:
                        client = service.video_editors.client
                        if (
                            client
                            and client.url == editor.server_url
                            and client.cancel_preparation(editor.preparation_request_id)
                        ):
                            with service.store.transaction() as session:
                                current = session.get(VideoEditor, editor.id)
                                session.update(
                                    current.model_copy(update={"preparation_request_id": None})
                                )
                    except Exception:
                        logger.exception(
                            "CVAT preparation stop unconfirmed; retaining visitor reservation"
                        )
            with service.store.transaction() as session:
                for token in session.list(LoginSession):
                    if token.user_id in expired and token.expires_at > now:
                        session.update(token.model_copy(update={"expires_at": now}))
                running = service.jobs.executing()
                for identifier in expired:
                    pending = any(
                        j.id in running
                        and (
                            j.user_id == identifier
                            or (j.user_id is None and j.project_id in projects)
                        )
                        for j in session.list(Job)
                    )
                    pending |= any(
                        d.user_id == identifier and not d.ended
                        for d in session.list(DesktopSession)
                    )
                    pending |= any(
                        e.preparation_request_id
                        and (
                            e.user_id == identifier
                            or (e.user_id is None and e.project_id in projects)
                        )
                        for e in session.list(VideoEditor)
                    )
                    pending |= service.assistants.running_for(identifier)
                    lease = session.get(DemoAccess, identifier)
                    if not pending and lease.stopped_at is None:
                        session.update(lease.model_copy(update={"stopped_at": now}))
                        service.audit.record("demo.expired", identifier)

    def plan(self, session: Session) -> dict[str, JsonValue]:
        leases = {d.id for d in session.list(DemoAccess)}
        users = sorted(u.id for u in session.list(User) if u.id in leases and not u.is_admin)
        projects = sorted(self.private_projects(session, set(users)))
        round_ = current_round(session)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "round": round_.id,
                    "users": users,
                    "projects": projects,
                    "members": [m.model_dump(mode="json") for m in session.list(Membership)],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return {
            "round_id": round_.id,
            "confirmation_token": fingerprint,
            "visitor_count": len(users),
            "project_count": len(projects),
            "user_ids": cast(list[JsonValue], users),
            "project_ids": cast(list[JsonValue], projects),
        }

    def preview(self) -> dict[str, JsonValue]:
        if not self.enabled:
            raise DomainError("Demo lifecycle controls are not enabled.", status=403)
        with self.services.store.transaction() as session:
            return self.plan(session)

    def reset(self, user: User, confirmation: str, token: str) -> dict[str, JsonValue]:
        if not self.enabled or not user.is_admin:
            raise DomainError(
                "An administrator must enable and confirm the demo reset.", status=403
            )
        if confirmation != "START FRESH DEMO":
            raise DomainError("Type START FRESH DEMO to confirm removal of demo visitors.")
        service = self.services
        with self.lock, service.store.transaction() as session:
            plan = self.plan(session)
            if token != plan["confirmation_token"]:
                raise Conflict("The demo changed. Review a fresh reset preview before confirming.")
            if service.jobs.executing() or any(
                j.status not in TERMINAL_STATUSES for j in session.list(Job)
            ):
                raise Conflict("Wait for all jobs to stop before resetting the demo.")
            if any(not d.ended for d in session.list(DesktopSession)) or service.assistants.active:
                raise Conflict(
                    "Close browser desktops and wait for assistant replies before resetting."
                )
            if any(e.preparation_request_id for e in session.list(VideoEditor)):
                raise Conflict("Wait for CVAT video preparation to stop before resetting.")
            users = set(cast(list[str], plan["user_ids"]))
            projects = set(cast(list[str], plan["project_ids"]))
            now = datetime.now(UTC)
            feedback_kinds = {"Conversation", "ToolExecution", "Job", "JobLog", "ReviewDecision"}
            all_rows = list(
                session.connection.execute("SELECT kind,id,project_id,data FROM records")
            )
            for kind, identifier, project_id, data in all_rows:
                content = json.loads(data)
                owned = project_id in projects or content.get("user_id") in users
                if owned and kind in feedback_kinds:
                    session.insert(
                        DemoFeedback(
                            round_id=str(plan["round_id"]),
                            source_kind=kind,
                            source_id=identifier,
                            content=content,
                        )
                    )
                    if project_id not in projects:
                        # Non-project chats belong to the removed identity, not the next visitor.
                        session.connection.execute(
                            "DELETE FROM records WHERE kind=? AND id=?", (kind, identifier)
                        )
            for project_id in projects:
                session.connection.execute(
                    "INSERT OR IGNORE INTO deleted_resources(kind,id) VALUES('Project',?)",
                    (project_id,),
                )
                Deletion.remove(session, Deletion.rows(session, project_id))
                session.connection.execute(
                    "DELETE FROM records WHERE kind='Project' AND id=?", (project_id,)
                )
            for identifier in users:
                old_user = session.get(User, identifier)
                lease = session.get(DemoAccess, identifier)
                visit(session, old_user, lease)
                identity = session.get(DemoVisit, identifier)
                session.update(identity.model_copy(update={"reset_at": now}))
                session.connection.execute(
                    "DELETE FROM records WHERE kind IN ('User','Password','DemoAccess') AND id=?",
                    (identifier,),
                )
                session.connection.execute(
                    "DELETE FROM records WHERE kind IN ('LoginSession','Membership') "
                    "AND json_extract(data,'$.user_id')=?",
                    (identifier,),
                )
            round_ = DemoRound()
            session.insert(round_)
            service.audit.record(
                "demo.reset",
                user.id,
                old_round_id=plan["round_id"],
                new_round_id=round_.id,
                removed_visitors=len(users),
                removed_projects=len(projects),
            )
        return {
            "round_id": round_.id,
            "removed_visitors": len(users),
            "removed_projects": len(projects),
            "feedback_preserved": True,
            "shared_caches_preserved": True,
        }
