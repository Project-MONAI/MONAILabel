from monai.utils import optional_import

from monailabel.config import settings


def is_sam2_module_available():
    # Global opt-out: skip SAM entirely (no download, no init) when set,
    # e.g. MONAI_LABEL_SKIP_SAM=True, instead of per-app `-c sam2 false`.
    if settings.MONAI_LABEL_SKIP_SAM:
        return False
    try:
        _, flag = optional_import("sam2")
        return flag
    except ImportError:
        return False
