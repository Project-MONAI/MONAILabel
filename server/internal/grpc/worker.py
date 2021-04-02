import argparse
import asyncio
import json
import logging
import logging.config
import os
import sys

import grpc

from server.interface import ServerException, ServerError, MONAIApp
from server.internal.grpc.protos import app_pb2, app_pb2_grpc
from server.utils.class_utils import get_class_of_subclass_from_file
from server.utils.generic import init_log_config

logger = logging.getLogger(__name__)


class AppService(app_pb2_grpc.AppServicer):

    def __init__(self, app, app_dir):
        logger.info(f"Initializing App:: {app} => {app_dir}")

        self.app = app
        self.app_dir = app_dir
        logger.info('Using app dir: {}'.format(app_dir))

        app_dir_lib = os.path.join(app_dir, 'lib')
        sys.path.append(app_dir)
        sys.path.append(app_dir_lib)

        main_py = os.path.join(self.app_dir, 'main.py')
        if not os.path.exists(main_py):
            raise ServerException(ServerError.APP_INIT_ERROR, f"App '{app}' Does NOT have main.py")

        # TODO:: Remove MONAIApp dependency and search for any class that has infer and train methods?
        c = get_class_of_subclass_from_file("main", main_py, MONAIApp)
        if c is None:
            raise ServerException(ServerError.APP_INIT_ERROR, f"App '{app}' Does NOT Implement MONAIApp in main.py")

        o = c(name=app, app_dir=self.app_dir)
        if not hasattr(o, "infer"):
            raise ServerException(
                ServerError.APP_INIT_ERROR,
                f"App '{app}' Does NOT Implement 'infer' method in main.py")
        if not hasattr(o, "train"):
            raise ServerException(
                ServerError.APP_INIT_ERROR,
                f"App '{app}' Does NOT Implement 'train' method in main.py")

        self.app_instance = o
        logger.info(f"{app} - Init Successful")

    async def RunInference(
            self,
            request: app_pb2.InferenceRequest,
            context: grpc.aio.ServicerContext) -> app_pb2.InferenceResponse:

        # TODO:: Process actual request
        request = {
            "image": "/workspace/Data/_image.nii.gz",
            "params": {}
        }

        result = self.app_instance.infer(request=request)
        return app_pb2.InferenceResponse() if result is None else app_pb2.InferenceResponse(
            label=result[0], params=json.dumps(result[1]))

    async def RunTraining(
            self,
            request: app_pb2.TrainRequest,
            context: grpc.aio.ServicerContext) -> app_pb2.TrainResponse:

        next_run_id = 0  # TODO:: Scan run directories and get the next one
        output_dir = os.path.join(self.app_dir, "model", f"run_{next_run_id}")

        # TODO:: Process actual request
        workspace = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'workspace'))
        dataset_root = os.path.join(workspace, "datasets", "Task09_Spleen")

        request = {
            'output_dir': output_dir,
            'data_list': os.path.join(dataset_root, "dataset.json"),
            'data_root': dataset_root,
            'device': "cuda",
            'epochs': 1,
            'amp': True,
            'train': {},
            'val': {},
        }

        result = self.app_instance.train(request=request)
        return app_pb2.TrainResponse(response=json.dumps(result) if not result else None)


async def serve(args) -> None:
    server = grpc.aio.server()
    app_pb2_grpc.add_AppServicer_to_server(AppService(args.name, args.path), server)

    listen_addr = '[::]:' + str(args.port)
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)

    await server.start()
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(0)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-a', '--path', required=True)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-p', '--port', default=50051, type=int)

    args = parser.parse_args()
    args.path = os.path.realpath(args.path)

    for arg in vars(args):
        logger.info('USING:: {} = {}'.format(arg, getattr(args, arg)))
    print("")

    logs_dir = os.path.join(args.path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_config = init_log_config(None, args.path, "app.log")
    if os.path.exists(log_config):
        with open(log_config, 'r') as f:
            logging.config.dictConfig(json.load(f))

    asyncio.run(serve(args))


if __name__ == '__main__':
    run_main()
