import os
import sys
import time
import uuid
from concurrent import futures
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+"/codes")
from grpc_server_util import *

import grpc

from GRPC_server.grpc_utils import save_chunks_to_file, get_file_chunks
from GRPC_server.resolution_pb2 import ProcessStatus, UploadStatus
from GRPC_server.resolution_pb2_grpc import SuperResolutionGRPCServicer, add_SuperResolutionGRPCServicer_to_server


class SuperResolutionGRPC(SuperResolutionGRPCServicer):
    def Upload(self, request_iterator, context):
        newfile = str(uuid.uuid4()) + ".mp4"
        save_chunks_to_file(request_iterator, "server_tmp_files/input/" + newfile)
        write_csv_new_line([newfile, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "None", "None"])
        return UploadStatus(status=1, file_key=newfile, error_msg="")

    def CheckProcess(self, request, context):
        find_filekey = request.file_key
        findline, queue = read_findkey_csv(find_filekey)
        if findline[0] != "None":
            if(findline[2]=="Error"):
                return ProcessStatus(status=11, message="File Processing Error")
            elif findline[2] == "None":
                return ProcessStatus(status=6, message="Process is in queue(" + str(queue) + ")")
            elif findline[2] == "Proc":
                return ProcessStatus(status=7, message="The file is now processing")
            elif findline[3] != "None":
                return ProcessStatus(status=9, message="File is deleted (old file key)")
            else:
                return ProcessStatus(status=8, message="Process completed (can download)")
        return ProcessStatus(status=10, message="Wrong key")

    def Download(self, request, context):
        return get_file_chunks("server_tmp_files/output/" + request.file_key)


def scheduler():
    find_proc_file()


def server_start(ip, port):
    initial_dir()
    initial_csv()
    schedule_count = 0

    print("server started >> " + ip + ":" + str(port))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_SuperResolutionGRPCServicer_to_server(SuperResolutionGRPC(), server)
    server.add_insecure_port(ip + ":" + str(port))
    server.start()
    try:
        while True:
            scheduler()
            schedule_count += 1
            if(schedule_count>17279):
                schedule_count = 0
                tmpfile_del()

            time.sleep(5)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    server_start("0.0.0.0", 50051)
