#TEMPLATE serving-eval as argus

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 as sdk

RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
    sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
    rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

RUN apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates

RUN cd /tmp && \
    wget http://devtools.dl.atlab.ai/aisdk/github_release/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh && \
    chmod +x cmake-3.13.2-Linux-x86_64.sh && \
    /tmp/cmake-3.13.2-Linux-x86_64.sh --skip-license --prefix=/usr

ARG PROTOBUF_VERSION=3.6.1
RUN cd /tmp && \
    wget http://devtools.dl.atlab.ai/aisdk/github_release/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz && \
    tar -zxvf protobuf-all-${PROTOBUF_VERSION}.tar.gz && \
    cd protobuf-${PROTOBUF_VERSION} && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install 

ENV TensorRT_FILE=TensorRT-4.0.1.6.Ubuntu-16.04.4.x86_64-gnu.cuda-9.0.cudnn7.1.tar.gz
RUN wget http://pbv7wun2s.bkt.clouddn.com/$TensorRT_FILE -O /tmp/$TensorRT_FILE && \
    tar xzf /tmp/$TensorRT_FILE -C /tmp && cd /tmp/TensorRT-4.0.1.6 && \
    cp include/Nv* /usr/local/include && cp -P lib/libnv* /usr/local/lib && ldconfig && \
    rm -rf /tmp/TensorRT-4.0.1.6

# opencv 3
RUN export OPENCV_ROOT=/tmp/opencv OPENCV_VER=3.4.1 && cd /tmp && \
    wget http://pbv7wun2s.bkt.clouddn.com/opencv-3.4.1.tar && tar -xvf opencv-3.4.1.tar && mv opencv-3.4.1 opencv && \
    mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/ \
    -D WITH_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=1 -D WITH_NVCUVID=on -D CUDA_GENERATION=Auto .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/* 

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgoogle-glog-dev \
    python-dev 

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.tar.gz && \
    tar -zxvf zeromq-4.2.5.tar.gz && \
    cd zeromq-4.2.5 && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install

ENV TRON_ROOT=/opt/tron
RUN mkdir TRON_ROOT
COPY cpp $TRON_ROOT/cpp
RUN mkdir $TRON_ROOT/cpp/build && \
    cd $TRON_ROOT/cpp/build && \
    cmake -DAPP=huanyuanka .. && \
    make -j"$(nproc)"

################################################################################

FROM sdk

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=lib2
ENV GOTRACEBACK=crash

WORKDIR /workspace/serving
RUN mkdir /tmp/eval

COPY --from=argus /go/bin/serving-eval /workspace/serving/serving-eval
COPY --from=sdk /opt/tron/cpp/build/lib/libinference_mix.so /workspace/serving/inference.so

ADD models.prototxt /workspace/serving/models.prototxt
ADD models /workspace/serving/models

CMD ["serving-eval","-f","serving-eval.conf"]