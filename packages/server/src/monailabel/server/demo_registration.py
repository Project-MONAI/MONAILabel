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

"""Small private-pilot starter project; metadata only, no downloads or inference."""

from monailabel.core.models import Label, Membership, ModelRecord, Project, Role
from monailabel.providers.spatial import MODELS as SPATIAL_MODELS
from monailabel.server.demo_gpt import ensure as ensure_gpt
from monailabel.server.storage import Session
from monailabel.totalsegmentator.catalog import MODELS as TOTAL_MODELS


def create_project(session: Session, user_id: str, *, restricted_roles: bool = False) -> Project:
    project = Project(
        name="My demo",
        labels=[Label(id=0, name="Background"), Label(id=1, name="Spleen")],
        instructions=(
            "Public demo data only. Load prepared radiology, pathology and endoscopy examples "
            "from Overview or Datasets. Use Slicer/OHIF for CT, QuPath for pathology and CVAT "
            "for video. SAM 2.1 uses points or a box for images, slices and video tracking; "
            "nnInteractive uses spatial prompts for CT/MRI volumes. Inspect and correct results "
            "before submitting for review. Training examples are recorded, not live training."
        ),
    )
    model = ModelRecord(
        project_id=project.id,
        name="VISTA3D",
        provider="vista3d",
        label_ids=[0],
        preset="vista3d",
        read_only=True,
    )
    project = project.model_copy(update={"annotation_model_id": model.id})
    session.insert(project)
    session.insert(model)
    for provider, (name, _) in TOTAL_MODELS.items():
        session.insert(
            ModelRecord(
                project_id=project.id,
                name=name,
                provider=provider,
                label_ids=[0],
                preset=provider,
                read_only=True,
            )
        )
    for provider, spec in SPATIAL_MODELS.items():
        session.insert(
            ModelRecord(
                project_id=project.id,
                name=spec.name,
                provider=provider,
                label_ids=[0],
                preset=provider,
                read_only=True,
            )
        )
    ensure_gpt(session, project.id)
    roles = [Role.ANNOTATOR, Role.REVIEWER] if restricted_roles else [Role.MANAGER]
    session.insert(Membership(project_id=project.id, user_id=user_id, roles=roles))
    return project
