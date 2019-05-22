FROM ubuntu:16.04
LABEL maintainer atlab@qiniu.com

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        vim \
        unzip \
        libprotobuf-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy
#    rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://pypi.douban.com/simple \
        numpy scipy sklearn   prometheus_client

RUN mkdir -p /workspace/serving

COPY evals /workspace/serving/python/evals
COPY build /workspace/serving/build
RUN cd /workspace/serving/build/py_eval && make && cd - \
    && mv /workspace/serving/build/py_eval/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py_eval/inference_pb2.py /workspace/serving/python/inference_pb2.py \
    && mv /workspace/serving/build/py_eval/inference.py /workspace/serving/python/inference.py 

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-face-cluster"