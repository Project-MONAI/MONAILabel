# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from collections import deque
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from monailabel.config import settings

router = APIRouter(
    prefix="/logs",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)

HTML_TEMPLATE = r"""
<html>
<head>
    <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.11/ace.js"></script>
    <script>
        $(document).ready(function () {
            fetch();
        });

        function fetch() {
            $.get("", { text: "true" }).done(function (data) {
                var editor = ace.edit("editor");
                editor.session.setValue(data);

                editor.setTheme("ace/theme/github");
                editor.session.setMode("ace/mode/matlab");

                editor.setReadOnly(true);
                editor.setOption("showLineNumbers", false);
                editor.setOption("showGutter", false);
                editor.setOption("showPrintMargin", false);
                editor.resize(true);
                editor.scrollToLine(editor.session.getLength(), true, true, function () {
                });
                editor.gotoLine(editor.session.getLength());
            });
        }
        REFRESH_T
    </script>
</head>

<body>
<div id="editor" style="height: 100%; font-size: medium">
</div>
</body>
</html>
"""


def get_logs(logger_file, lines, html, text, refresh):
    if not os.path.isfile(logger_file):
        raise HTTPException(status_code=404, detail=f"Log File {logger_file} NOT Found")

    refresh = max(refresh, 3) if refresh else 0
    if lines > 0:
        with open(logger_file) as fin:
            response_lines = list(deque(fin, lines))
            if html and not text:
                response = HTML_TEMPLATE.replace("LINES_T", str(lines))
                response = response.replace(
                    "REFRESH_T",
                    "setInterval(fetch, 1000*" + str(refresh) + ");" if refresh else "",
                )
                response_type = "text/html"
            else:
                response = "".join(response_lines)
                response_type = "text/plain"
            return Response(content=response, media_type=response_type)
    return FileResponse(logger_file, media_type="text/plain")


@router.get("/", summary="Get Logs")
async def api_get_logs(
    logfile: Optional[str] = "app.log",
    lines: Optional[int] = 300,
    html: Optional[bool] = True,
    text: Optional[bool] = False,
    refresh: Optional[int] = 0,
):
    return get_logs(os.path.join(settings.MONAI_LABEL_APP_DIR, "logs", str(logfile)), lines, html, text, refresh)


@router.get("/gpu", summary="Get GPU Info (nvidia-smi)")
async def gpu_info():
    response = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")
    return Response(content=response, media_type="text/plain")
