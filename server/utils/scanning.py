import json
import logging
import os

from server.core.config import settings

logger = logging.getLogger(__name__)


def scan_apps():
    apps_dir = os.path.join(settings.WORKSPACE, "apps")
    apps = dict()
    for f in os.scandir(apps_dir):
        if f.is_dir():
            meta_file = os.path.join(f.path, 'meta.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as fc:
                    meta = json.load(fc)
                apps[f.name] = {"path": f.path, "meta": meta}
            else:
                logger.warning(f"{f.name} exists but meta.json is missing.  App is not ready yet!")
    return apps


def init_apps():
    apps = scan_apps()
    for app in apps:
        print(f"init {app}")


if __name__ == '__main__':
    init_apps()
