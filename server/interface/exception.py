# TODO:: Better Name ??

class ServerError:
    SERVER_ERROR = "SERVER_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    CLASS_INIT_ERROR = "CLASS_INIT_ERROR"
    MODEL_IMPORT_ERROR = "MODEL_IMPORT_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"

    APP_NOT_FOUND = "APP_NOT_FOUND"
    APP_INIT_ERROR = "APP_INIT_ERROR"
    APP_INFERENCE_FAILED = "APP_INFERENCE_FAILED"
    APP_TRAIN_FAILED = "APP_TRAIN_FAILED"
    APP_ERROR = "APP_ERROR"


class ServerException(Exception):
    def __init__(self, error, msg):
        self.error = error
        self.msg = msg
