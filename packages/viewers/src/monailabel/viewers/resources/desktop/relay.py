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

"""Expose one mounted Unix gateway on the network-isolated container's loopback."""

import asyncio
from contextlib import suppress

SOCKET = "/gateway/api.sock"
PORT = 8765


class Relay:
    def __init__(self) -> None:
        self.connections = 0

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.connections >= 4:
            writer.close()
            return
        self.connections += 1
        upstream: asyncio.StreamWriter | None = None
        tasks: list[asyncio.Task[None]] = []

        async def copy(source: asyncio.StreamReader, target: asyncio.StreamWriter) -> None:
            while chunk := await source.read(65536):
                target.write(chunk)
                await target.drain()

        try:
            async with asyncio.timeout(180):
                remote, upstream = await asyncio.open_unix_connection(SOCKET)
                tasks = [
                    asyncio.create_task(copy(reader, upstream)),
                    asyncio.create_task(copy(remote, writer)),
                ]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (OSError, TimeoutError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for stream in (writer, upstream):
                if stream is not None:
                    stream.close()
                    with suppress(OSError):
                        await stream.wait_closed()
            self.connections -= 1


async def main() -> None:
    relay = Relay()
    server = await asyncio.start_server(relay.handle, "127.0.0.1", PORT, backlog=4)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
