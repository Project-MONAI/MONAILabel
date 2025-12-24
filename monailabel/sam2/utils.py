from monai.utils import optional_import


def is_sam2_module_available():
    try:
        _, flag = optional_import("sam2")
        return flag
    except ImportError:
        return False
