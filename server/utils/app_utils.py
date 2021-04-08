import logging
import os

from server.core.config import settings
from server.interface import MONAIApp
from server.interface.exception import ServerError, ServerException
from server.utils.class_utils import get_class_of_subclass_from_file

logger = logging.getLogger(__name__)
app_instance = None


def get_app_instance():
    global app_instance
    if app_instance is not None:
        return app_instance

    app_dir = settings.APP_DIR
    logger.info(f"Initializing App from: {app_dir}")

    main_py = os.path.join(app_dir, 'main.py')
    if not os.path.exists(main_py):
        raise ServerException(ServerError.APP_INIT_ERROR, f"App Does NOT have main.py")

    c = get_class_of_subclass_from_file("main", main_py, MONAIApp)
    if c is None:
        raise ServerException(ServerError.APP_INIT_ERROR, f"App Does NOT Implement MONAIApp in main.py")

    o = c(app_dir=app_dir)
    methods = ["infer", "train", "info", "stop_train", "next_sample", "save_label"]
    for m in methods:
        if not hasattr(o, m):
            raise ServerException(ServerError.APP_INIT_ERROR, f"App Does NOT Implement '{m}' method in main.py")

    app_instance = o
    return app_instance
