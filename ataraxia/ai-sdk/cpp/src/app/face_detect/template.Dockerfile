#TEMPLATE serving-eval as argus

FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04 as sdk

RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
        sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
        rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        curl \
        vim \
        git \
        ca-certificates \
        libatlas-base-dev \
        libboost-all-dev \
        libgoogle-glog-dev

RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8

RUN cd /tmp && wget http://devtools.dl.atlab.ai/aisdk/github_release/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh && chmod +x cmake-3.13.2-Linux-x86_64.sh && /tmp/cmake-3.13.2-Linux-x86_64.sh --skip-license --prefix=/usr

# opencv 3.4.1
RUN export OPENCV_ROOT=/tmp/opencv && cd /tmp && \
        wget http://devtools.dl.atlab.ai/aisdk/other/from_tensorrt/opencv-3.4.1.tar && tar -xvf opencv-3.4.1.tar && mv opencv-3.4.1 opencv && \
        mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
        cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/ \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_opencv_apps=OFF \
        -D WITH_PROTOBUF=OFF \
        -DBUILD_LIST=core,highgui,improc,imgcodecs,videoio,cudev,cudaimgproc \
        -D WITH_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=1 -D WITH_NVCUVID=on -D CUDA_GENERATION=Auto .. && \
        make -j"$(nproc)" && make install && ldconfig

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.tar.gz && \
        tar -zxvf zeromq-4.2.5.tar.gz && \
        cd zeromq-4.2.5 && \
        ./configure --prefix=/usr && make -j"$(nproc)" && make install

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz && \
        tar -zxvf protobuf-all-3.6.1.tar.gz && \
        cd protobuf-3.6.1 && \
        ./configure --prefix=/usr && make -j"$(nproc)" && make install 

LABEL com.qiniu.atlab.os = "ubuntu-16.04"

ENV TRON_ROOT=/opt/tron
RUN mkdir TRON_ROOT
COPY cpp $TRON_ROOT/cpp
RUN mkdir $TRON_ROOT/cpp/build && \
        cd $TRON_ROOT/cpp/build && \
        cmake -DAPP=face_detect .. && \
        make -j"$(nproc)"

################################################################################

FROM sdk

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=lib2

WORKDIR /workspace/serving
RUN mkdir /tmp/eval

COPY --from=argus /go/bin/serving-eval /workspace/serving/serving-eval
COPY --from=sdk /opt/tron/cpp/build/lib/libinference_fd.so /workspace/serving/inference.so

ADD models.prototxt /workspace/serving/models.prototxt
ADD models /workspace/serving/models

CMD ["serving-eval","-f","serving-eval.conf"]
