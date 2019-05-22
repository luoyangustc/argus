FROM nvidia/cuda:8.0-cudnn6-devel
LABEL maintainer "nathangq <nathangq@gmail.com>"

RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    zip \
    unzip \
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
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN mkdir -p /opt && cd /opt && wget http://devtools.dl.atlab.ai/aisdk/other/RefineDet.tar.gz && tar -zxvf RefineDet.tar.gz && mv RefineDet caffe

ENV CAFFE_ROOT=/opt/caffe/
WORKDIR $CAFFE_ROOT

# Missing "packaging" package
RUN python -m pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip && \
    pip install -i https://mirrors.aliyun.com/pypi/simple packaging


RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN cd python && for req in $(cat requirements.txt) pydot; do pip install -i https://mirrors.aliyun.com/pypi/simple $req; done && cd ..

RUN cp Makefile.config.example Makefile.config && sed -i "s:/usr/lib/python2.7:/usr/local/lib/python2.7:g" Makefile.config
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim
#RUN mkdir build && cd build && \
#    cmake -DUSE_CUDNN=1 .. && \
#    WITH_PYTHON_LAYER=1 make -j"$(nproc)" && make pycaffe
RUN make all -j"$(nproc)" && make py

# Set the environment variables so that the paths are correctly configured
ENV PYTHONPATH $CAFFE_ROOT/python:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$CAFFE_ROOT/python:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

RUN pip install -i https://mirrors.aliyun.com/pypi/simple numpy==1.16.1 && \
    pip install -i https://mirrors.aliyun.com/pypi/simple matplotlib==2.2.3 && \
    pip install -i https://mirrors.aliyun.com/pypi/simple scikit-image && \
    pip install -i https://mirrors.aliyun.com/pypi/simple protobuf && \
    pip install -i https://mirrors.aliyun.com/pypi/simple jieba && \
    pip install -i https://mirrors.aliyun.com/pypi/simple opencv-python-headless==4.0.0.21 && python -c 'import cv2;assert cv2.__version__[0]=="4"' && \
    ldconfig


WORKDIR /workspace

# 上面的是 FROM reg-xs.qiniu.io/atlab/zhatu:testv1-180318 的内容
# https://github.com/qbox/ataraxia/blob/dev/dockerfiles/terror/python/terror-classify/Dockerfile#L1
