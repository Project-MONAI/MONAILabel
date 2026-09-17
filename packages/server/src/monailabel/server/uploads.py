"""Bound binary request bodies while receiving them, including chunked uploads."""

import io

from fastapi import Request

from monailabel.core.errors import DomainError
from monailabel.server.data import MAX_FILE_BYTES


async def read_upload(request: Request) -> bytes:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError as exc:
            raise DomainError("Invalid Content-Length.", status=400) from exc
        if size < 0:
            raise DomainError("Invalid Content-Length.", status=400)
        if size > MAX_FILE_BYTES:
            raise DomainError("Image file exceeds 128 MiB.", status=413)
    with io.BytesIO() as stream:
        async for block in request.stream():
            if stream.tell() + len(block) > MAX_FILE_BYTES:
                raise DomainError("Image file exceeds 128 MiB.", status=413)
            stream.write(block)
        if not stream.tell():
            raise DomainError("Select a nonempty image file.")
        return stream.getvalue()
