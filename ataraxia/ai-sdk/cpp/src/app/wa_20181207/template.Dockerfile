#TEMPLATE serving-eval as argus

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 as sdk

RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && sed -i "s:security.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list 
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt install -y cmake g++ gdb valgrind aria2 unzip

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

ENV TensorRT_FILE=TensorRT-4.0.1.6.Ubuntu-16.04.4.x86_64-gnu.cuda-9.0.cudnn7.1.tar.gz
RUN wget http://devtools.dl.atlab.ai/aisdk/other/from_tensorrt/$TensorRT_FILE -O /tmp/$TensorRT_FILE && \
    tar xzf /tmp/$TensorRT_FILE -C /tmp && cd /tmp/TensorRT-4.0.1.6 && \
    cp include/Nv* /usr/local/include && cp -P lib/libnv* /usr/local/lib && ldconfig && \
    rm -rf /tmp/TensorRT-4.0.1.6

# opencv 3
RUN export OPENCV_ROOT=/tmp/opencv OPENCV_VER=3.4.1 && cd /tmp && \
    wget http://devtools.dl.atlab.ai/aisdk/other/from_tensorrt/opencv-3.4.1.tar && tar -xvf opencv-3.4.1.tar && mv opencv-3.4.1 opencv && \
    mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/ \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D WITH_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=1 -D WITH_NVCUVID=on -D CUDA_GENERATION=Auto .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/* 

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.tar.gz && \
    tar -zxvf zeromq-4.2.5.tar.gz && \
    cd zeromq-4.2.5 && \
    ./configure --prefix=/usr && make -j"${nproc}" && make install

RUN wget http://devtools.dl.atlab.ai/aisdk/github_release/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz && \
    tar -zxvf protobuf-all-3.6.1.tar.gz && \
    cd protobuf-3.6.1 && \
    ./configure --prefix=/usr && make -j"$(nproc)" && make install 


RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libgoogle-glog-dev \
    libboost-all-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ENV MIX_ROOT=/opt/mix
# RUN mkdir ${MIX_ROOT}
# COPY Tron-mixup $MIX_ROOT/Tron-mixup
# RUN mkdir $MIX_ROOT/Tron-mixup/mixup/build && \
#     cd $MIX_ROOT/Tron-mixup/mixup/build && \
#     cmake .. && \
#     make -j"$(nproc)"
COPY libwangan.so /usr/local/lib/libwangan.so

ENV TRON_ROOT=/opt/tron
RUN mkdir TRON_ROOT
COPY cpp $TRON_ROOT/cpp
RUN mkdir $TRON_ROOT/cpp/build && \
    cd $TRON_ROOT/cpp/build && \
    cmake -DAPP=wa_20181207 .. && \
    make -j"$(nproc)"
RUN rm -rf $TRON_ROOT/cpp/src

################################################################################

FROM sdk

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=lib2

WORKDIR /workspace/serving

COPY --from=argus /go/bin/serving-eval /workspace/serving/serving-eval
COPY --from=sdk /opt/tron/cpp/build/lib/libinference_wa.so /workspace/serving/inference.so

CMD ["serving-eval","-f","serving-eval.conf"]