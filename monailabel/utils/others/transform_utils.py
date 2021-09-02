# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

from monai.transforms import Compose

from monailabel.interfaces.exception import MONAILabelError, MONAILabelException

logger = logging.getLogger(__name__)


def dump_data(data):
    if logging.getLogger().level == logging.DEBUG:
        logger.debug("**************************** DATA ********************************************")
        for k in data:
            v = data[k]
            logger.debug(
                "Data key: {} = {}".format(
                    k,
                    v.shape
                    if hasattr(v, "shape")
                    else v
                    if type(v) in (int, float, bool, str, dict, tuple, list)
                    else type(v),
                )
            )
        logger.debug("******************************************************************************")


def shape_info(data, keys=("image", "label", "logits", "pred", "model")):
    info = []
    for key in keys:
        val = data.get(key) if hasattr(data, "get") else None
        if val is not None and hasattr(val, "shape"):
            info.append("{}: {}({})".format(key, val.shape, val.dtype))
    return "; ".join(info)


def run_transforms(data, callables, inverse=False, log_prefix="POST", log_name="Transform"):
    """
    Run Transforms

    :param data: Input data dictionary
    :param callables: List of transforms or callable objects
    :param inverse: Run inverse instead of call/forward function
    :param log_prefix: Logging prefix (POST or PRE)
    :param log_name: Type of callables for logging
    :return: Processed data after running transforms
    """
    logger.info("{} - Run {}".format(log_prefix, log_name))
    logger.info("{} - Input Keys: {}".format(log_prefix, data.keys()))

    if not callables:
        return data

    compose = Compose()
    if isinstance(callables, Compose):
        callables = callables.transforms
    elif callable(callables):
        callables = [callables]

    for t in callables:
        name = t.__class__.__name__
        start = time.time()

        dump_data(data)
        if inverse:
            if hasattr(t, "inverse"):
                data = t.inverse(data)
            else:
                raise MONAILabelException(
                    MONAILabelError.TRANSFORM_ERROR,
                    "{} '{}' has no invert method".format(log_name, t.__class__.__name__),
                )
        elif callable(t):
            compose.transforms = [t]
            data = compose(data)
        else:
            raise MONAILabelException(
                MONAILabelError.TRANSFORM_ERROR,
                "{} '{}' is not callable".format(log_name, t.__class__.__name__),
            )

        logger.info(
            "{} - {} ({}): Time: {:.4f}; {}".format(
                log_prefix,
                log_name,
                name,
                float(time.time() - start),
                shape_info(data),
            )
        )
        logger.debug("-----------------------------------------------------------------------------")

    dump_data(data)
    return data
