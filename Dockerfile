FROM ubuntu:20.04
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends build-essential
FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

#RUN apt-get install -y python3.7 python3-pip
#RUN ln -s /usr/bin/python3 /usr/bin/python


# Install dependencies:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get install -y git
RUN git clone https://github.com/bigscience-workshop/promptsource.git
RUN cd promptsource && pip install -r requirements.txt && pip install -e .

RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip uninstall protobuf
RUN pip install --no-binary protobuf protobuf

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
COPY scripts/download_t0.py .
RUN python download_t0.py

COPY . .
