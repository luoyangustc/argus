#TEMPLATE serving-eval as argus

FROM ubuntu:16.04 as sdk

LABEL com.qiniu.atlab.os = "ubuntu-16.04"

RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list 
RUN sed -i "s:security.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    vim \
    git \
    ca-certificates \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    python-dev \
    python-numpy \
    python-pip \
    python-setuptools

RUN cd /tmp && wget http://devtools.dl.atlab.ai/aisdk/github_release/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh && chmod +x cmake-3.13.2-Linux-x86_64.sh && /tmp/cmake-3.13.2-Linux-x86_64.sh --skip-license --prefix=/usr

RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz && \
    tar -zxvf protobuf-all-3.6.1.tar.gz && \
    cd protobuf-3.6.1 && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install 

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

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.tar.gz && \
    tar -zxvf zeromq-4.2.5.tar.gz && \
    cd zeromq-4.2.5 && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install

ENV TRON_ROOT=/opt/tron
RUN mkdir TRON_ROOT
COPY cpp $TRON_ROOT/cpp
RUN mkdir $TRON_ROOT/cpp/build && \
    cd $TRON_ROOT/cpp/build && \
    cmake -DAPP=face_search .. && \
    make -j"$(nproc)"

################################################################################

FROM sdk

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=lib2

WORKDIR /workspace/serving
RUN mkdir /tmp/eval

COPY --from=argus /go/bin/serving-eval /workspace/serving/serving-eval
COPY --from=sdk /opt/tron/cpp/build/lib/libinference_fs.so /workspace/serving/inference.so

CMD ["serving-eval","-f","serving-eval.conf"]
