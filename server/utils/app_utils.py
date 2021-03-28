import json
import logging
import mimetypes
import os
import sys

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from requests_toolbelt import MultipartEncoder

from server.interface import MONAIApp
from server.utils.class_utils import get_class_of_subclass_from_file
from server.utils.scanning import scan_apps

logger = logging.getLogger(__name__)


def remove_file(path: str) -> None:
    os.unlink(path)


def remove_path(paths):
    for p in paths:
        sys.path.remove(p)


# TODO:: Get it done through GRPC
def get_app_instance(app, background_tasks):
    apps = scan_apps()
    if app not in apps:
        raise HTTPException(status_code=404, detail=f"App '{app}' NOT Found")

    app_dir = apps[app]['path']
    app_dir_lib = os.path.join(app_dir, 'lib')
    logger.info('Using app dir: {}'.format(app_dir))

    sys.path.append(app_dir)
    sys.path.append(app_dir_lib)
    background_tasks.add_task(remove_path, [app_dir_lib, app_dir])

    main_py = os.path.join(app_dir, 'main.py')
    if not os.path.exists(main_py):
        raise HTTPException(status_code=404, detail=f"App '{app}' Does NOT have main.py")

    c = get_class_of_subclass_from_file("main", main_py, MONAIApp)
    if c is None:
        raise HTTPException(status_code=404, detail=f"App '{app}' Does NOT Implement MONAIApp in main.py")

    o = c(name=app, app_dir=app_dir)
    if not hasattr(o, "infer"):
        raise HTTPException(status_code=404, detail=f"App '{app}' Does NOT Implement 'infer' method in main.py")
    return o, apps[app]


def send_response(app, result, output, background_tasks):
    if result is None:
        raise HTTPException(status_code=500, detail=f"Failed to execute infer for {app}")

    res_img, res_json = result
    if res_img is None or output == 'json':
        return res_json

    background_tasks.add_task(remove_file, res_img)
    m_type = mimetypes.guess_type(res_img, strict=False)
    logger.debug(f"Guessed Mime Type for Image: {m_type}")

    if m_type is None or m_type[0] is None:
        m_type = "application/octet-stream"
    else:
        m_type = f"{m_type[0]}/{m_type[1]}"
    logger.debug(f"Final Mime Type: {m_type}")

    if res_json is None or not len(res_json) or output == 'image':
        return FileResponse(res_img, media_type=m_type)

    res_fields = dict()
    res_fields['params'] = (None, json.dumps(res_json), 'application/json')
    res_fields['image'] = (os.path.basename(res_img), open(res_img, 'rb'), m_type)

    return_message = MultipartEncoder(fields=res_fields)
    return Response(content=return_message.to_string(), media_type=return_message.content_type)
