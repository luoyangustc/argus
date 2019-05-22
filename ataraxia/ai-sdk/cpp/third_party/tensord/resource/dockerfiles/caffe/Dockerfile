ARG CUDA_VERSION=8.0
ARG CUDNN_VERSION=6
ARG UBUNTU_VERSION=16.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    sed -i "s:security.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

#TEMPLATE tensord_base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libopenblas-dev \
    libhdf5-serial-dev \
    liblmdb-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libatlas-base-dev \
    python-dev \
    python-numpy

#install Caffe
RUN cd /tmp && \
    wget http://devtools.dl.atlab.ai/aisdk/github_release/NVIDIA/nccl/releases/download/v2.4.2/nccl.tar.gz && \
    tar xzvf nccl.tar.gz && cd nccl && \
    make -j"$(nproc)" install && \
    cd .. && rm -rf nccl && \
    wget http://devtools.dl.atlab.ai/aisdk/github/BVLC/caffe/caffe.tar.gz && \
    tar -zxvf caffe.tar.gz && cd caffe && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DCMAKE_CXX_FLAGS="-std=c++11" .. && \
    make -j"$(nproc)" && make install && \
    cp -rf install/include/caffe /usr/local/include/ && \
    cp -rf install/lib/* /usr/local/lib/ && \
    cd /tmp && rm -rf caffe
