FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    vim \
    git \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cd /tmp && wget http://devtools.dl.atlab.ai/aisdk/github_release/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh && chmod +x cmake-3.13.2-Linux-x86_64.sh && /tmp/cmake-3.13.2-Linux-x86_64.sh --skip-license --prefix=/usr

# RUN export OPENCV_CONTRIB_ROOT=/tmp/opencv-contrib OPENCV_ROOT=/tmp/opencv OPENCV_VER=3.2.0 && \
# git clone -b ${OPENCV_VER} --depth 1 https://github.com/opencv/opencv.git ${OPENCV_ROOT} && \
# git clone -b ${OPENCV_VER} --depth 1 https://github.com/opencv/opencv_contrib.git ${OPENCV_CONTRIB_ROOT} && \
# mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
# cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
# -D OPENCV_ICV_URL="http://devtools.dl.atlab.ai/docker/" \
# -D OPENCV_PROTOBUF_URL="http://devtools.dl.atlab.ai/docker/" \
# -D OPENCV_CONTRIB_BOOSTDESC_URL="http://devtools.dl.atlab.ai/docker/" \
# -D OPENCV_CONTRIB_VGG_URL="http://devtools.dl.atlab.ai/docker/" \
# -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF \
# -D OPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_ROOT}/modules \
# -D WITH_CUDA=OFF -D BUILD_opencv_python2=ON -D BUILD_EXAMPLES=OFF .. && \
# make -j"$(nproc)" && make install && ldconfig && \
# rm -rf /tmp/*
RUN export OPENCV_CONTRIB_ROOT=/tmp/opencv-contrib OPENCV_ROOT=/tmp/opencv OPENCV_VER=3.2.0 && \
    cd /tmp && \
    wget http://devtools.dl.atlab.ai/aisdk/other/from_tensorrt/opencv-3.4.1.tar && \
    tar -xvf opencv-3.4.1.tar && mv opencv-3.4.1 opencv && \
    mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_ICV_URL="http://devtools.dl.atlab.ai/docker/" \
    -D OPENCV_PROTOBUF_URL="http://devtools.dl.atlab.ai/docker/" \
    -D OPENCV_CONTRIB_BOOSTDESC_URL="http://devtools.dl.atlab.ai/docker/" \
    -D OPENCV_CONTRIB_VGG_URL="http://devtools.dl.atlab.ai/docker/" \
    -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D WITH_CUDA=OFF -D BUILD_opencv_python2=ON -D BUILD_EXAMPLES=OFF .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/*

RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.tar.gz && \
    tar -zxvf zeromq-4.2.5.tar.gz && \
    cd zeromq-4.2.5 && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz && \
    tar -zxvf protobuf-all-3.6.1.tar.gz && \
    cd protobuf-3.6.1 && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install 

LABEL com.qiniu.atlab.os = "ubuntu-16.04"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libsnappy-dev \
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
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DCMAKE_CXX_FLAGS="-std=c++11" .. && \
    make -j"$(nproc)"

# ENV PYCAFFE_ROOT $CAFFE_ROOT/python
# ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
# ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
#-------------- make and install caffe

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#install mxnet

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs
RUN ln -s /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so \
    /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so.1

RUN wget http://devtools.dl.atlab.ai/aisdk/other/incubator-mxnet-1.3.1.tar.gz && tar -zxvf incubator-mxnet-1.3.1.tar.gz && mv incubator-mxnet mxnet && \
    cd mxnet && \
    make -j"$(nproc)" USE_CPP_PACKAGE=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda-8.0

# RUN cp -rf $CAFFE_ROOT/mxnet/lib/* /usr/local/lib/
RUN cp -rf `pwd`/mxnet/lib/* /usr/local/lib/
RUN cp -rf `pwd`/mxnet/include/mxnet /usr/local/include/

RUN cd build && make install
RUN cp -rf build/install/include/caffe /usr/local/include/
RUN cp -rf $CAFFE_ROOT/build/install/lib/* /usr/local/lib/

RUN python -m pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip
RUN pip install -i https://mirrors.aliyun.com/pypi/simple opencv-python
# ENV PYTHONPATH=/opt/caffe/mxnet/python:$PYTHONPATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    valgrind && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ENV TRON_ROOT=/opt/tron
# RUN cmkdir TRON_ROOT
# COPY cpp $TRON_ROOT/cpp
# RUN mkdir $TRON_ROOT/cpp/build && \
#     cd $TRON_ROOT/cpp/build && \
#     cmake -DCMAKE_BUILD_TYPE=DEBUG -DAPP=face_feature .. && \
#     make -j"$(nproc)" && \
#     valgrind --tool=memcheck --leak-check=full ./bin/test_ff
