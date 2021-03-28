import json
import logging
import os

from fastapi import APIRouter, HTTPException

from server.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"],
    responses={404: {"description": "Not found"}},
)


# TODO:: Define template for Dataset Folder? Or Dataset JSON?


def scan_datasets():
    datasets_dir = os.path.join(settings.WORKSPACE, "datasets")
    datasets = dict()
    for f in os.scandir(datasets_dir):
        if f.is_dir():
            meta_file = os.path.join(f.path, 'dataset.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as fc:
                    meta = json.load(fc)
                datasets[f.name] = {"path": f.path, "meta": meta}
            else:
                logger.warning(f"{f.name} exists but dataset.json is missing.  Dataset is not ready yet!")
    return datasets


@router.get("/", summary="List All Datasets available")
async def get_datasets():
    datasets = scan_datasets()
    return [{
        'name': k,
        'training': len(v.get('training', [])),
        'validation': len(v.get('validation', [])),
        'testing': len(v.get('testing', []))
    } for k, v in datasets.items()]


@router.get("/{dataset}", summary="List More details about an existing Dataset")
async def get_dataset(dataset: str):
    datasets = scan_datasets()
    if dataset not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' NOT Found")
    return datasets[dataset]["meta"]


@router.put("/{dataset}", summary="Upload a new Dataset")
async def create_dataset(dataset: str):
    raise HTTPException(status_code=501, detail=f"Not Implemented")


@router.delete("/{dataset}", summary="Delete an existing Dataset")
async def delete_dataset(dataset: str):
    datasets = scan_datasets()
    if dataset not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' NOT Found")
    raise HTTPException(status_code=501, detail=f"Not Implemented")
