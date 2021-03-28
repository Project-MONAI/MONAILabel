from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/session",
    tags=["Session"],
    responses={404: {"description": "Not found"}},
)

# TODO:: Do we need session support?  Useful for inference (if external image is loaded out of dataset)

fake_sessions_db = {"1": {"name": "XYZ"}, "2": {"name": "ABC"}}


@router.get("/{session}", summary="Get Details about an existing session")
async def get_session(session: str):
    if session not in fake_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    return fake_sessions_db[session]["name"]


@router.put("/{session}", summary="Create a new Session")
async def create_session(session: str):
    fake_sessions_db[session] = "DEF"
    return fake_sessions_db[session]


@router.patch("/{session}", summary="Add more data to an existing Session")
async def update_session(session: str):
    fake_sessions_db[session] = "DEF"
    return fake_sessions_db[session]


@router.delete("/{session}", summary="Delete an existing session")
async def delete_session(session: str):
    if session not in fake_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    return fake_sessions_db.pop(session)
