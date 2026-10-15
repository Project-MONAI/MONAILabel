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

"""Load a private host-owned OpenAI key after the local-only demo wrapper.

Default mode checks file safety only: no API request or service change. Use
--run followed by a command only in an approved private test or deployment.
Never pass an API key on the command line.
"""

import argparse
import os
import resource
import stat
from pathlib import Path


def read_key(path: Path) -> str:
    if not path.is_absolute():
        raise ValueError("Use an absolute private key-file path.")
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("The key directory must be owned by this user and private (0700).")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "r", encoding="ascii") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
        ):
            raise ValueError("The key must be a private, user-owned regular file (0600).")
        raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError("The private key file is too large.")
        value = raw.strip()
    if not (
        20 <= len(value) <= 4096
        and value.startswith("sk-")
        and all(33 <= ord(c) <= 126 for c in value)
    ):
        raise ValueError("The private file does not contain a valid API-key format.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--run", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    try:
        key = read_key(args.key_file)
    except (OSError, ValueError, UnicodeError):
        raise SystemExit(
            "Key check failed. Check the private file, ownership and permissions."
        ) from None
    if not args.run:
        print("Private key file ready; no API request made and no service changed.")
        return
    environment = os.environ | {"OPENAI_API_KEY": key}
    os.execvpe(args.run[0], args.run, environment)


if __name__ == "__main__":
    main()
