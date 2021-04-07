import logging
import os
import threading

from server.interface import ServerException, ServerError
from server.utils.generic import run_command
from server.utils.scanning import scan_apps

logger = logging.getLogger(__name__)


def app_info(app):
    apps = scan_apps()
    if app not in apps:
        raise ServerException(ServerError.APP_NOT_FOUND, f"{app} Not Found")
    return apps[app]


def init_app(app, app_dir, port=0):
    class AppT(threading.Thread):
        def __init__(self, name, path):
            threading.Thread.__init__(self)

            self.cmd = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'internal', 'grpc', 'worker.sh'))
            self.name = name
            self.path = path
            self.port = port

        def run(self):
            logger.info(f"Init App: {self.name}; path: {self.path}")
            run_command(self.cmd, [self.name, self.path, self.port], logging.getLogger(self.name))

    t = AppT(app, app_dir)
    t.start()
    return t


def init_apps():
    apps = scan_apps()
    return [init_app(app, apps[app]['path']) for counter, app in enumerate(apps)]


def get_grpc_port(app):
    info = app_info(app)
    port_file = os.path.join(info['path'], '.port')

    if os.path.isfile(port_file):
        with open(port_file, 'r') as f:
            return int(f.read())
    return None


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    try:
        app_threads = init_apps()
        [t.join() for t in app_threads]
    except KeyboardInterrupt:
        exit(0)
