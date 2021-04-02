import asyncio
import json
import logging
import os

import grpc

from server.internal.grpc.protos import app_pb2, app_pb2_grpc

# For more channel options, please see https://grpc.io/grpc/core/group__grpc__arg__keys.html
CHANNEL_OPTIONS = [
    ('grpc.lb_policy_name', 'pick_first'),
    ('grpc.enable_retries', 0),
    ('grpc.keepalive_timeout_ms', 30000)
]

logger = logging.getLogger(__name__)


async def inference(image, params, port=50051) -> None:
    async with grpc.aio.insecure_channel(
            target='localhost:' + str(port),
            options=CHANNEL_OPTIONS) as channel:
        stub = app_pb2_grpc.AppStub(channel)

        # Timeout in seconds.
        # Please refer gRPC Python documents for more detail. https://grpc.io/grpc/python/grpc.html
        request = app_pb2.InferenceRequest(image=image, params=json.dumps(params))
        response = await stub.RunInference(request, timeout=30)
    logger.info("Label: {}".format(response.label))
    logger.info("Params: {}".format(json.loads(response.params) if response.params else None))

    if os.path.exists(response.label):
        os.unlink(response.label)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    image = '/workspace/Data/_image.nii.gz'
    params = {}
    asyncio.run(inference(image, params))
