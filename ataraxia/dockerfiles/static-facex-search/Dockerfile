# TODO replace by image in cs hub
FROM nvidia/cuda:8.0-cudnn6-devel

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install  -y --no-install-recommends  libprotobuf-dev libsnappy-dev libopenblas-dev protobuf-compiler \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake \
    python-dev python-pip && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean

# pip
RUN  pip  --no-cache-dir install -U  setuptools -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com && \
     pip --no-cache-dir install hickle nose pylint pyyaml numpy nose-timer requests protobuf easydict prometheus_client  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

RUN export CUDAMAT_VERSION=master CUDAMAT_ROOT=/opt/cudamat && \
    git clone -b ${CUDAMAT_VERSION} --depth 1 https://github.com/cudamat/cudamat.git ${CUDAMAT_ROOT} && \
    cd ${CUDAMAT_ROOT} && pip install . && \
    mv /usr/local/lib/python2.7/dist-packages/cudamat/*.so $CUDAMAT_ROOT/cudamat/

RUN mkdir -p /workspace/serving

ADD evals /workspace/serving/python/evals
ADD build /workspace/serving/build
RUN cd /workspace/serving/build/py_eval && make && cd -
RUN mv /workspace/serving/build/py_eval/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py_eval/inference_pb2.py /workspace/serving/python/inference_pb2.py 
RUN mv /workspace/serving/build/py_eval/inference_py.py /workspace/serving/python/inference_py.py 

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:/opt/cudamat:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-pulp"