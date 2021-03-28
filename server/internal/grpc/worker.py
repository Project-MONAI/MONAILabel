import asyncio
import logging

import grpc

from server.internal.grpc.protos import app_pb2, app_pb2_grpc

# TODO:: This way for each app/pipeline we can init a seperate process+vnenv
#  and worker app will execute actions like train/inference etc.. in a seperate process
logger = logging.getLogger(__name__)


class AppService(app_pb2_grpc.AppServicer):

    async def RunInference(
            self,
            request: app_pb2.InferenceRequest,
            context: grpc.aio.ServicerContext) -> app_pb2.InferenceResponse:
        # Run Inference...
        response = app_pb2.InferenceResponse(label='xyz.nii.gz', params='{}')
        return response

    async def RunTraining(
            self,
            request: app_pb2.TrainRequest,
            context: grpc.aio.ServicerContext) -> app_pb2.TrainResponse:
        # Run Training...
        response = app_pb2.TrainResponse(response='{}')
        return response


async def serve(port=50051) -> None:
    server = grpc.aio.server()
    app_pb2_grpc.add_GreeterServicer_to_server(AppService(), server)

    listen_addr = '[::]:' + str(port)
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)

    await server.start()
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
