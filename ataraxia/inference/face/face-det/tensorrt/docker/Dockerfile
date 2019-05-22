FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer = "Qiniu ATLab<ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    curl \
    vim \
    git \
    ca-certificates \
    libprotobuf-dev \
    protobuf-compiler && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


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


########## INSTALLATION STEPS ###################
ENV TRON_ROOT=/opt/tron
RUN mkdir TRON_ROOT
COPY . $TRON_ROOT
RUN sh $TRON_ROOT/scripts/build_shell.sh && \
    cp $TRON_ROOT/build/lib/libinference.so /usr/lib && \
    cp $TRON_ROOT/build/lib/libfacedetection.so /usr/lib && \
    mkdir -p /workspace/serving && \
    ln -s /usr/lib/libinference.so /workspace/serving/inference.so && \
    rm -rf $TRON_ROOT


RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
LABEL com.qiniu.atlab.os = "ubuntu-16.04"
LABEL com.qiniu.atlab.type = "tron"
