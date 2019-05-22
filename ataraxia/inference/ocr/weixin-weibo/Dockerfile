FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER Elaine Bao <elaine.bao@hotmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
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

# opencv 3
RUN export OPENCV_CONTRIB_ROOT=/tmp/opencv-contrib OPENCV_ROOT=/tmp/opencv OPENCV_VER=3.2.0 && \
    git clone -b ${OPENCV_VER} --depth 1 https://github.com/opencv/opencv.git ${OPENCV_ROOT} && \
    git clone -b ${OPENCV_VER} --depth 1 https://github.com/opencv/opencv_contrib.git ${OPENCV_CONTRIB_ROOT} && \
    mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_ICV_URL="http://devtools.dl.atlab.ai/docker/" \
    -D OPENCV_PROTOBUF_URL="http://devtools.dl.atlab.ai/docker/" \
    -D OPENCV_CONTRIB_BOOSTDESC_URL="http://devtools.dl.atlab.ai/docker/" \
    -D OPENCV_CONTRIB_VGG_URL="http://devtools.dl.atlab.ai/docker/" \
    -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_ROOT}/modules \
    -D WITH_CUDA=ON -D BUILD_opencv_python2=ON -D BUILD_EXAMPLES=OFF .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/*

WORKDIR /opt
RUN git clone https://github.com/ElaineBao/CTPN.git && \
    mv CTPN/caffe /opt/ && \
    rm -rf CTPN/

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# Missing "packaging" package
RUN pip install --upgrade pip && \
    pip install packaging

RUN cd python && for req in $(cat requirements.txt) pydot; do pip install -i https://pypi.tuna.tsinghua.edu.cn/simple $req; done && cd ..

RUN cp Makefile.config.example Makefile.config
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim
RUN mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 .. && \
    WITH_PYTHON_LAYER=1 make -j"$(nproc)" && make pycaffe

# Set the environment variables so that the paths are correctly configured
ENV PYTHONPATH $CAFFE_ROOT/python:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$CAFFE_ROOT/python:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numpy && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple protobuf && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jieba && \
    ldconfig

RUN wget http://p1wqzrl8v.bkt.clouddn.com/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
RUN pip install torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl && \
    rm torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl && \
    pip install torchvision

WORKDIR /workspace
