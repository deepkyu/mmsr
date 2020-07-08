FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

ENV CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get -y upgrade && apt-get -y install libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg
RUN conda install -y cudatoolkit cudnn
COPY ./requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install grpcio --ignore-installed
RUN python3 -m pip install grpc-tools==0.0.1
# set TORCH_CUDA_ARCH_LIST properly. (6, 1) -> 6.1
EXPOSE 50051
ARG TORCH_CUDA_ARCH_LIST=7.0
COPY . /workspace/mmsr
WORKDIR /workspace/mmsr
RUN cd ./codes/models/archs/dcn && python3 setup.py develop
WORKDIR /workspace/mmsr
ENTRYPOINT python3 GRPC_server/grpc_server.py
