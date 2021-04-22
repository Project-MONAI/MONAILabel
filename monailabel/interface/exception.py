class MONAILabelError:
    """
    Attributes:
        SERVER_ERROR -            Server Error
        UNKNOWN_ERROR -           Unknown Error
        CLASS_INIT_ERROR -        Class Initialization Error
        MODEL_IMPORT_ERROR -      Model Import Error
        INFERENCE_ERROR -         Inference Error

        APP_INIT_ERROR -          Initialization Error
        APP_INFERENCE_FAILED -    Inference Failed
        APP_TRAIN_FAILED -        Train Failed
        APP_ERROR APP -           General Error
    """

    SERVER_ERROR = "SERVER_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    CLASS_INIT_ERROR = "CLASS_INIT_ERROR"
    MODEL_IMPORT_ERROR = "MODEL_IMPORT_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"

    APP_INIT_ERROR = "APP_INIT_ERROR"
    APP_INFERENCE_FAILED = "APP_INFERENCE_FAILED"
    APP_TRAIN_FAILED = "APP_TRAIN_FAILED"
    APP_ERROR = "APP_ERROR"


class MONAILabelException(Exception):
    """
    MONAI Label Exception
    """

    def __init__(self, error: MONAILabelError, msg: str):
        self.error = error
        self.msg = msg

class UnknownLabelIdException(MONAILabelException):

    def __init__(self, msg: str):
        super().__init__(MONAILabelException.APP_ERROR, msg)
