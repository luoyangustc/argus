FROM registry.docker-cn.com/nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
RUN cat /etc/apt/sources.list && ls -alh /etc/apt/sources.list.d && rm /etc/apt/sources.list.d/*.list
RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim \
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
        python-tk \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip numpy==1.15.0
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple prometheus_client Shapely Flask matplotlib==2.1.2 scipy plumbum numpy ipython==5.0 Pillow protobuf pycuda tensorflow-gpu==1.4.0 cython opencv-python
RUN mkdir -p /workspace/serving/python/evals

ADD build /workspace/serving/build
ADD evals /workspace/serving/python/evals
ADD src /workspace/serving/python/evals/src
RUN cd /workspace/serving/build/py && make && cd -
RUN mv /workspace/serving/build/py/inference.so /workspace/serving/inference.so && mv /workspace/serving/build/py/inference_pb2.py /workspace/serving/python/inference_pb2.py
RUN mv /workspace/serving/build/py/inference_py.py /workspace/serving/python/inference_py.py 

RUN mkdir -p /workspace/tensorrt_package && cd /workspace/tensorrt_package && wget http://pco79ka6h.bkt.clouddn.com/TensorRT-4.0.1.6.Ubuntu-16.04.4.x86_64-gnu.cuda-8.0.cudnn7.1.tar.gz && tar xvf TensorRT-4.0.1.6.Ubuntu-16.04.4.x86_64-gnu.cuda-8.0.cudnn7.1.tar.gz && cd TensorRT-4.0.1.6 && cd python && pip install tensorrt-4.0.1.6-cp27-cp27mu-linux_x86_64.whl && cd ../graphsurgeon/ && pip install graphsurgeon-0.2.0-py2.py3-none-any.whl && cd ../uff/ && pip install uff-0.4.0-py2.py3-none-any.whl

RUN cd /usr/lib/x86_64-linux-gnu/ && ln -s libcudnn.so.7 libcudnn.so.6
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/workspace/tensorrt_package/TensorRT-4.0.1.6/lib:/usr/local/nvidia/lib64/
ENV PYTHONPATH=/workspace/serving/python/evals/src:/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-ocr-detection-general"
