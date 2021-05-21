import importlib.util
import inspect


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_class_of_subclass(module, class_c):
    for _, o in inspect.getmembers(module):
        if inspect.isclass(o) and o != class_c and issubclass(o, class_c):
            return o
    return None


def get_class_of_subclass_from_file(module_name, file_path, class_c):
    return get_class_of_subclass(module_from_file(module_name, file_path), class_c)
