ARG CUDA_VER=8.0
ARG CUDNN_VER=6
ARG UBUNTU_VER=16.04
FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-devel-ubuntu${UBUNTU_VER}

ENV PATH /usr/local/cuda/bin:$PATH
ENV LC_ALL=C.UTF-8

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    sed -i "s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# TODO 待细化
RUN apt-get install -y --no-install-recommends \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake \
    python-dev python-pip python-tk \
    libatlas-base-dev libopencv-dev libcurl4-openssl-dev \
    libgtest-dev \
    openssh-server rsync && \
    cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG OPENCV_VER=3.4.1
RUN export OPENCV_ROOT=/tmp/opencv && \
    cd /tmp && wget http://pbv7wun2s.bkt.clouddn.com/opencv-3.4.1.tar && tar -xvf opencv-3.4.1.tar && mv opencv-3.4.1 opencv && \
    mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D WITH_CUDA=ON -D BUILD_opencv_python2=ON -D BUILD_EXAMPLES=OFF .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/*

# TODO 待细化
RUN python2 -m pip --no-cache-dir --default-timeout=6000 install --index-url https://mirrors.aliyun.com/pypi/simple -U pip setuptools && \
    python2 -m pip --no-cache-dir --default-timeout=6000 install --index-url https://mirrors.aliyun.com/pypi/simple \
    hickle nose pyyaml numpy nose-timer requests easydict matplotlib==2.2.3 cython scikit-image docopt httplib2

ARG MXNET_VER=1.5.0b20190220
RUN python2 -m pip --no-cache-dir install --upgrade -i https://mirrors.aliyun.com/pypi/simple mxnet-cu80==${MXNET_VER}

################################################################################

RUN python2 -m pip --no-cache-dir install --upgrade -i https://mirrors.aliyun.com/pypi/simple \
    scikit-learn python-opencv
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.6  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN export http_proxy= && \
    export https_proxy= && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python2 get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py
RUN python3 -m pip --no-cache-dir --default-timeout=6000 install --index-url https://mirrors.aliyun.com/pypi/simple \
    jupyterlab

# config ssh server
RUN mkdir /var/run/sshd && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    echo "export LC_ALL=C.UTF-8" >> /etc/profile

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini
ENTRYPOINT ["/bin/tini", "--"]
