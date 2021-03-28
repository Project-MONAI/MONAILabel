from fastapi import APIRouter

router = APIRouter(
    prefix="/activelearning",
    tags=["AppEngine"],
    responses={404: {"description": "Not found"}},
)


@router.post("/{app}", summary="Run Active Learning strategy for an existing App")
async def run_active_learning(app: str):
    return {"app": app}
