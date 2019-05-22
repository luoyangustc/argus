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
    libopenblas-dev \
    python-dev

#install Mxnet
ARG MXNET_VERSION=1.3.1
RUN ln -s /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so \
    /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so.1
RUN cd /tmp && \
    wget http://devtools.dl.atlab.ai/aisdk/other/incubator-mxnet-${MXNET_VERSION}.tar.gz && \
    tar -zxvf incubator-mxnet-${MXNET_VERSION}.tar.gz && \
    mv incubator-mxnet mxnet && \
    cd mxnet && \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs \
    make -j"$(nproc)" USE_CPP_PACKAGE=1 USE_BLAS=openblas USE_CUDA=1 USE_OPENCV=0 USE_CUDA_PATH=/usr/local/cuda-8.0 && \
    cp -rf `pwd`/lib/* /usr/local/lib/ && \
    cp -rf `pwd`/include/mxnet /usr/local/include/ && \
    rm -rf mxnet
