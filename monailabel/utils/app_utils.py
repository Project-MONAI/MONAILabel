import logging
import os

from monailabel.core.config import settings
from monailabel.interface import MONAILabelApp
from monailabel.interface.exception import MONAILabelError, MONAILabelException
from monailabel.utils.class_utils import get_class_of_subclass_from_file

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
        raise MONAILabelException(MONAILabelError.APP_INIT_ERROR, f"App Does NOT have main.py")

    c = get_class_of_subclass_from_file("main", main_py, MONAILabelApp)
    if c is None:
        raise MONAILabelException(MONAILabelError.APP_INIT_ERROR, f"App Does NOT Implement MONAILabelApp in main.py")

    o = c(app_dir=app_dir)
    methods = ["infer", "train", "info", "next_sample", "save_label"]
    for m in methods:
        if not hasattr(o, m):
            raise MONAILabelException(MONAILabelError.APP_INIT_ERROR, f"App Does NOT Implement '{m}' method in main.py")

    app_instance = o
    return app_instance
