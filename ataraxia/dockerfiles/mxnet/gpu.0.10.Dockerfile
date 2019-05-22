FROM reg.qiniu.com/ava-public/ava-cuda8-cudnn6:1.0
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
# 这两个 NVIDIA source list 更新存在问题
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# apt-get && python && pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake \
    python-dev python-pip python-tk \
    libatlas-base-dev libopencv-dev libcurl4-openssl-dev \
    libgtest-dev \
    openssh-server rsync && \
    cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# pip
RUN pip --no-cache-dir --default-timeout=6000 install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -U pip setuptools && \
    pip --no-cache-dir --default-timeout=6000 install --index-url https://pypi.tuna.tsinghua.edu.cn/simple hickle nose pylint pyyaml numpy nose-timer requests easydict matplotlib cython scikit-image

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

# fcis
ENV FCIS_ROOT=/opt/fcis
ENV PYFCIS_ROOT=$FCIS_ROOT/fcis:$FCIS_ROOT/lib
RUN git clone -b master --depth 1 https://github.com/msracver/FCIS.git ${FCIS_ROOT} && \
    cd $FCIS_ROOT && sh init.sh

# mxnet
ENV MXNET_ROOT=/opt/mxnet MXNET_VER=v0.10.0-support
RUN mkdir -p ${MXNET_ROOT} && cd ${MXNET_ROOT} && git clone -b ${MXNET_VER} --depth 1 --recursive https://github.com/ataraxialab/incubator-mxnet.git . && \
    cp ${FCIS_ROOT}/fcis/operator_cxx/* ${MXNET_ROOT}/src/operator/contrib -r && \
    pip install -U pip setuptools && pip install nose pylint numpy nose-timer requests && \
    make -j"$(nproc)" USE_DIST_KVSTORE=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 \
    EXTRA_OPERATORS=${MXNET_ROOT}/example/rcnn/operator && \
    rm -rf build && cd ${MXNET_ROOT}/example/rcnn && make

ENV PYTHONPATH $MXNET_ROOT/python:$PYFCIS_ROOT:$PYTHONPATH
ENV PATH /usr/local/cuda/bin:$PATH


RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple ava-sdk

# 将时区改成 GMT+8
RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
LABEL com.qiniu.atlab.os = "ubuntu-16.04"
LABEL com.qiniu.atlab.type = "mxnet"
