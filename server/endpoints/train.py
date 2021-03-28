import logging
import os

from fastapi import APIRouter
from starlette.background import BackgroundTasks

from server.core.config import settings
from server.utils.app_utils import get_app_instance
from server.utils.generic import run_command

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/train",
    tags=["AppEngine"],
    responses={404: {"description": "Not found"}},
)


@router.post("/{app}", summary="Run Train action for an existing App")
async def run_train(app: str, background_tasks: BackgroundTasks):
    instance, app_info = get_app_instance(app, background_tasks)

    # TODO:: Fix this.. some weird issue wrt Exception: Can't pickle (multi-processing related in dataloader)
    if 1 == 1:
        run_command(f"python {app_info['path']}/test.py train")
        return {"training": "success"}

    app_dir = app_info['path']
    output_dir = os.path.join(app_dir, "model", "run_0")
    dataset_root = os.path.join(settings.WORKSPACE, "datasets", "Task09_Spleen")

    request = {
        'output_dir': output_dir,
        'data_list': os.path.join(dataset_root, "dataset.json"),
        'data_root': dataset_root,
        'device': "cuda",
        'epochs': 1,
        'amp': True,
        'train': {},
        'val': {},
    }

    return instance.train(request=request)
