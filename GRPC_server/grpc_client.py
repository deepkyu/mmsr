import time

import grpc

from GRPC_server.grpc_utils import get_file_chunks, save_chunks_to_file
from GRPC_server.resolution_pb2 import KeyMessage
from GRPC_server.resolution_pb2_grpc import SuperResolutionGRPCStub

channel = grpc.insecure_channel('ip:port')
stub = SuperResolutionGRPCStub(channel)


def Upload(filepath):
    file_chunk_generator = get_file_chunks(filepath)
    response = stub.Upload(file_chunk_generator)
    return [response.status, response.file_key]


def CheckProcess(key):
    response = stub.CheckProcess(KeyMessage(file_key=key))
    return [response.status, response.message]


def Download(key, filepath):
    response = stub.Download(KeyMessage(file_key=key))
    save_chunks_to_file(response, filepath)


if __name__ == '__main__':
    # status_ret = Upload("/home/won/Desktop/inputex.mp4")
    # print(status_ret)
    # now_KEY = status_ret[1]
    # print("File KEY -> " + now_KEY)
    # while True:
    #     nowstatus = CheckProcess(now_KEY)
    #     print(nowstatus)
    #     time.sleep(5)
    #     if nowstatus[0] == 8:
    #         break
    # Download(now_KEY,"aa.mp4")
    print(CheckProcess("59fb4cd9-6d68-467f-81d4-4301dc8c3941.mp4"))