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

"""Fixed, cached examples for an admitted visitor, never arbitrary imports."""

from threading import Lock
from typing import Literal

from monailabel.core.dataset_templates import DatasetTemplateImport
from monailabel.core.errors import DomainError
from monailabel.core.models import Asset, Job, JobStatus, Label, Project, User
from monailabel.core.video import VideoAsset
from monailabel.server.dataset_templates import DatasetTemplates
from monailabel.server.demo_access import require_active
from monailabel.server.demo_context import actor
from monailabel.server.storage import Store

Example = Literal["radiology", "mri", "pathology", "endoscopy"]
TEMPLATES: dict[Example, str] = {
    "radiology": "Task09_Spleen",
    "mri": "prostate-mri",
    "pathology": "openslide-cmu-small",
    "endoscopy": "hyperkvasir-tool-tracking",
}


class DemoOnboarding:
    def __init__(self, store: Store, templates: DatasetTemplates):
        self.store, self.templates = store, templates
        self.lock = Lock()

    def start(self, user: User, example: Example = "radiology") -> Job | None:
        if example not in TEMPLATES:
            raise DomainError("Choose a prepared CT, MRI, pathology or endoscopy example.")
        lease = require_active(self.store, user)
        if not lease or not lease.restricted_roles or not lease.initial_project_id:
            return None
        if not lease.started_at:
            raise DomainError("Enter the demo before preparing its sample.", status=403)
        project_id = lease.initial_project_id
        with self.lock:
            if self.present(project_id, example):
                return None
            imports = [
                job for job in self.store.list(Job, project_id) if job.kind == "dataset_import"
            ]
            template_id = TEMPLATES[example]
            previous = [
                job
                for job in imports
                if job.request.get("template_id", "Task09_Spleen") == template_id
            ]
            for job in imports:
                if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    if job in previous:
                        return job
                    raise DomainError(
                        "Wait for the current sample preparation to finish.", status=409
                    )
            if len(previous) >= 3:
                raise DomainError("The host must check this demo's sample preparation.", status=409)
            token = actor.set(user.id)
            try:
                if example in {"pathology", "mri"}:
                    # Add one fixed target without renumbering or replacing existing labels.
                    # Video import already adds its source-declared Snare target.
                    with self.store.transaction() as session:
                        project = session.get(Project, project_id)
                        target = "Nucleus" if example == "pathology" else "Prostate"
                        if not any(
                            label.name.casefold() == target.casefold() for label in project.labels
                        ):
                            used = {label.id for label in project.labels}
                            identifier = next((i for i in range(1, 256) if i not in used), None)
                            if identifier is None:
                                raise DomainError("This project has no room for another label.")
                            session.update(
                                project.model_copy(
                                    update={
                                        "labels": [
                                            *project.labels,
                                            Label(id=identifier, name=target),
                                        ],
                                        "version": project.version + 1,
                                    }
                                )
                            )
                return self.templates.start(
                    project_id,
                    DatasetTemplateImport(template_id=template_id, limit=1, include_masks=False),
                    user.id,
                )
            finally:
                actor.reset(token)

    def present(self, project_id: str, example: Example) -> bool:
        if example == "endoscopy":
            return any(
                item.group_id == "hyperkvasir:99b387e7-d07b-4268-9226-4df450c2a198"
                for item in self.store.list(VideoAsset, project_id)
            )
        assets = self.store.list(Asset, project_id)
        if example == "pathology":
            return any(item.group_id == "openslide:CMU-1" for item in assets)
        prefix = "msd:Task05_Prostate:" if example == "mri" else "msd:Task09_Spleen:"
        return any(item.group_id.startswith(prefix) for item in assets)
