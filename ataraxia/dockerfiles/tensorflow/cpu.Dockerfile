FROM reg.qiniu.com/ava-public/ava-ubuntu1604:1.0
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

# apt-get && python && pip
# tensorflow install without gpu support

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake \
    python-dev python-pip python-tk \
    libatlas-base-dev libopencv-dev libcurl4-openssl-dev \
    libcurl3-dev \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    zip \
    zlib1g-dev \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    openssh-server rsync && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# pip
RUN pip --no-cache-dir --default-timeout=6000 install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -U pip setuptools && \
    pip --no-cache-dir --default-timeout=6000 install --index-url https://pypi.tuna.tsinghua.edu.cn/simple nose pylint numpy nose-timer requests easydict matplotlib cython scikit-image \
    flask future graphviz hypothesis jupyter protobuf pyyaml scipy setuptools  sklearn pandas \
    && \
    python -m ipykernel.kernelspec


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
    -D BUILD_opencv_python2=ON -D BUILD_EXAMPLES=OFF .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/*

# Set up our notebook config.
COPY tensorflow/jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY tensorflow/run_jupyter.sh /

# Set up Bazel.
# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc

# Install the most recent bazel release.
ENV BAZEL_VERSION 0.8.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl http://xsio.qiniu.io/bazel-0.8.0-installer-linux-x86_64.sh -H 'Host:otr41gcz3.bkt.clouddn.com' -o bazel-0.8.0-installer-linux-x86_64.sh &&\
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh


########## INSTALLATION STEPS ###################
ENV TENSORFLOW_ROOT=/opt/tensorflow TENSORFLOW_VER=master CI_BUILD_PYTHON=python
RUN mkdir -p ${TENSORFLOW_ROOT} && cd ${TENSORFLOW_ROOT} && git clone --branch ${TENSORFLOW_VER} --depth 1  --recursive https://github.com/ataraxialab/tensorflow.git . && \
    pip install wheel && \
    tensorflow/tools/ci_build/builds/configured CPU \
    bazel build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        # For optimized builds appropriate for the hardware platform of your choosing, uncomment below...
        # For ivy-bridge or sandy-bridge
        # --copt=-march="ivybridge" \
        # for haswell, broadwell, or skylake
        # --copt=-march="haswell" \
        tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /tmp/pip && \
    rm -rf /root/.cache
# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple ava-sdk
# 将时区改成 GMT+8
RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
LABEL com.qiniu.atlab.os = "ubuntu-16.04"
LABEL com.qiniu.atlab.type = "tensorflow"
