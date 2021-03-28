from fastapi import APIRouter

router = APIRouter(
    prefix="/logs",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)


# TODO:: Define Log Config for both Server and App
@router.get("/", summary="Get Current Server Logs")
async def get_logs():
    return "current server logs... (Last N lines from logfile)"


@router.get("/{app}", summary="Get Logs specific to an App")
async def get_app_logs(app: str):
    return f"current {app} logs... (Last N lines from logfile)"
