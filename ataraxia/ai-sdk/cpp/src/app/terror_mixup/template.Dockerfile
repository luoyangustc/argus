#TEMPLATE serving-eval as argus

FROM nvidia/cuda:8.0-cudnn6-devel as sdk
LABEL maintainer "nathangq <nathangq@gmail.com>"

RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends \
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
    libsnappy-dev \
    python-dev \
    python-numpy \
    python-pip \
    python-setuptools \
    python-scipy \
    apt-utils

RUN cd /tmp && wget http://devtools.dl.atlab.ai/aisdk/github_release/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh && chmod +x cmake-3.13.2-Linux-x86_64.sh && /tmp/cmake-3.13.2-Linux-x86_64.sh --skip-license --prefix=/usr

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz && \
    tar -zxvf protobuf-all-3.6.1.tar.gz && \
    cd protobuf-3.6.1 && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install 

RUN mkdir -p /opt && cd /opt && wget http://devtools.dl.atlab.ai/aisdk/other/RefineDet.tar.gz && tar -zxvf RefineDet.tar.gz && mv RefineDet caffe
ENV CAFFE_ROOT=/opt/caffe/
RUN cd /opt/caffe/ && rm CMakeLists.txt && wget http://devtools.dl.atlab.ai/aisdk/other/RefineDet-CMakeLists/CMakeLists.txt && mkdir build && cd build && cmake -DBUILD_docs=OFF -DUSE_OPENCV=OFF ..
RUN cd /opt/caffe/build && make -j"$(nproc)"
RUN cd /opt/caffe/build && make -j"$(nproc)" install
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# 上面的是 FROM reg-xs.qiniu.io/atlab/zhatu:testv1-180318 的内容
# https://github.com/qbox/ataraxia/blob/dev/dockerfiles/terror/python/terror-classify/Dockerfile#L1


RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.tar.gz && \
    tar -zxvf zeromq-4.2.5.tar.gz && \
    cd zeromq-4.2.5 && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install 
RUN ldconfig

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


ENV TRON_ROOT=/opt/tron
RUN mkdir TRON_ROOT
COPY cpp $TRON_ROOT/cpp
RUN mkdir $TRON_ROOT/cpp/build && \
    cd $TRON_ROOT/cpp/build && \
    cmake -DAPP=terror_mixup .. && \
    make -j"$(nproc)" inference_terror_mixup

################################################################################

FROM sdk

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=lib2
ENV GOTRACEBACK=crash

WORKDIR /workspace/serving
RUN mkdir /tmp/eval

COPY --from=argus /go/bin/serving-eval /workspace/serving/serving-eval
COPY --from=sdk /opt/tron/cpp/build/lib/libinference_terror_mixup.so /workspace/serving/inference.so

CMD ["serving-eval","-f","serving-eval.conf"]
