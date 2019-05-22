# TODO replace by image in cs hub
# FROM reg.qiniu.com/ava-public/atlab-base-caffe-gpu:ava
# FROM ava-caffe-deluxe.vedio.classify.senet:20180110-v59-vedio_classfiy

FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
LABEL maintainer atlab@qiniu.com

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
# 这两个 NVIDIA source list 更新存在问题
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# apt-get && python && pip

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#-------------- make and install caffe
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=1.0

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req \
    --index-url https://mirrors.ustc.edu.cn/pypi/web/simple; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
#-------------- make and install caffe

RUN pip install prometheus_client easydict protobuf

#install mxnet
RUN pip install mxnet-cu80==1.1.0 --index-url https://mirrors.ustc.edu.cn/pypi/web/simple

RUN apt-get update && apt-get install -y --no-install-recommends vim unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install opencv_python>=3.1.0 --index-url https://mirrors.ustc.edu.cn/pypi/web/simple

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
LABEL com.qiniu.atlab.biz = "ava-eval-face-feature"