FROM nvidia/cuda:8.0-cudnn6-devel

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



WORKDIR /opt
RUN git clone https://github.com/nathangq/RefineDet.git && \
    mkdir /opt/caffe && \
    mv RefineDet/* /opt/caffe/ && \
    rm -rf RefineDet/

ENV CAFFE_ROOT=/opt/caffe/
WORKDIR $CAFFE_ROOT

# Missing "packaging" package
RUN pip install --upgrade pip

RUN    pip install packaging


RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN cd python && for req in $(cat requirements.txt) pydot; do pip install -i https://pypi.tuna.tsinghua.edu.cn/simple $req; done && cd ..

RUN cp Makefile.config.example Makefile.config
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim
#RUN mkdir build && cd build && \
#    cmake -DUSE_CUDNN=1 .. && \
#    WITH_PYTHON_LAYER=1 make -j"$(nproc)" && make pycaffe

RUN make all -j"$(nproc)" && make py

# Set the environment variables so that the paths are correctly configured
ENV PYTHONPATH $CAFFE_ROOT/python:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$CAFFE_ROOT/python:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numpy
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib==2.1.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image==0.13.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple protobuf
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jieba
RUN pip --default-timeout=100 install opencv-python
RUN ldconfig

RUN mkdir -p /workspace/serving/python/evals

RUN pip install prometheus_client

ADD build /workspace/serving/build
ADD evals /workspace/serving/python/evals
RUN cd /workspace/serving/build/py && make && cd ~
RUN mv /workspace/serving/build/py/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py/inference_pb2.py /workspace/serving/python/inference_pb2.py
RUN mv /workspace/serving/build/py/inference_py.py /workspace/serving/python/inference_py.py


ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-terror-predetect-refinedet"
