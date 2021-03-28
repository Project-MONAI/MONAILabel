# TODO:: Better Name ??

class ServerError:
    SESSION_EXPIRED = "SESSION_EXPIRED"
    RESULT_NOT_FOUND = "RESULT_NOT_FOUND"
    SERVER_ERROR = "SERVER_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    MODEL_IMPORT_ERROR = "MODEL_IMPORT_ERROR"
    SERVER_CONFIG_ERROR = "SERVER_CONFIG_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    CLASS_INIT_ERROR = "CLASS_INIT_ERROR"


class ServerException(Exception):
    def __init__(self, error, msg):
        self.error = error
        self.msg = msg
