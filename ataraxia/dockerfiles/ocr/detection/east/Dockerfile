FROM reg.qiniu.com/inference/origin-xyf-bk-scene:20182337
RUN cat /etc/apt/sources.list && ls -alh /etc/apt/sources.list.d && rm /etc/apt/sources.list.d/*.list
RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

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
    
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple prometheus_client Shapely Flask matplotlib scipy plumbum numpy ipython Pillow protobuf
RUN mkdir -p /workspace/serving/python/evals

ADD build /workspace/serving/build
ADD evals /workspace/serving/python/evals
ADD src /workspace/serving/python/evals/src
RUN cd /workspace/serving/build/py && make && cd -
RUN mv /workspace/serving/build/py/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py/inference_pb2.py /workspace/serving/python/inference_pb2.py 
RUN mv /workspace/serving/build/py/inference_py.py /workspace/serving/python/inference_py.py 

ENV PYTHONPATH=/workspace/serving/python/evals/src:/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-ocr-detection-general"
