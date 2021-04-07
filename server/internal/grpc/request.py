import asyncio
import json
import logging

import grpc

from server.internal.grpc.protos import app_pb2, app_pb2_grpc

# For more channel options, please see https://grpc.io/grpc/core/group__grpc__arg__keys.html
CHANNEL_OPTIONS = [
    ('grpc.lb_policy_name', 'pick_first'),
    ('grpc.enable_retries', 0),
    ('grpc.keepalive_timeout_ms', 30000)
]

logger = logging.getLogger(__name__)


async def grpc_inference(request, port, timeout=30) -> None:
    async with grpc.aio.insecure_channel(target=f"localhost:{port}", options=CHANNEL_OPTIONS) as channel:
        stub = app_pb2_grpc.AppStub(channel)

        response = await stub.RunInference(app_pb2.Request(request=json.dumps(request)), timeout=timeout)
        return json.loads(response.response) if response else None


async def grpc_train(request, port, timeout=None) -> None:
    async with grpc.aio.insecure_channel(target=f"localhost:{port}", options=CHANNEL_OPTIONS) as channel:
        stub = app_pb2_grpc.AppStub(channel)

        response = await stub.RunTraining(app_pb2.Request(request=json.dumps(request)), timeout=timeout)
        return json.loads(response.response) if response else None


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    request = {
        "image": '/workspace/Data/_image.nii.gz',
        "params": {}
    }
    asyncio.run(grpc_inference(request, 50051))
