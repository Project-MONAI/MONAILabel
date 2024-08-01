import os
import logging
from typing import Any, Dict, Tuple

from monai.utils import ImageMetaKey
from monailabel.tasks.infer.bundle import BundleInferTask

logger = logging.getLogger(__name__)


class VISTAInfer(BundleInferTask):
    """
    This provides Inference Engine for pre-trained VISTA segmentation model.
    """
    def writer(self, data: Dict[str, Any], extension=None, dtype=None) -> Tuple[Any, Any]:
        d = dict(data)
        output_dir = self.bundle_config.get_parsed_content("output_dir", instantiate=True)
        output_ext = self.bundle_config.get_parsed_content("output_ext", instantiate=True)
        image_key = self.bundle_config.get_parsed_content("image_key", instantiate=True)
        output_postfix = self.bundle_config.get_parsed_content("output_postfix", instantiate=True)

        img = d.get(image_key, None)
        filename = img.meta.get(ImageMetaKey.FILENAME_OR_OBJ) if img is not None else None
        basename = os.path.splitext(os.path.basename(filename))[0] if filename else "mask"
        output_filename = f"{basename}{'_' + output_postfix if output_postfix else ''}{output_ext}"
        output_filepath = os.path.join(output_dir, output_filename)
        if os.path.exists(output_filepath):
            logger.info(f"Reusing the bundle output {output_filepath}.")
            return output_filepath, {}
        else:
            super().writer(data=data, extension=extension, dtype=dtype)