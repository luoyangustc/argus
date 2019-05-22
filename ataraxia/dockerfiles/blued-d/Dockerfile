# TODO replace by image in cs hub
FROM reg.qiniu.com/avaprd/mxnet:1.0.0

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

RUN apt-get update && apt-get install  -y --no-install-recommends  libprotobuf-dev libsnappy-dev libopenblas-dev protobuf-compiler && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean

RUN mkdir -p /workspace/serving

RUN pip install prometheus_client && pip install easydict && pip install protobuf pycrayon

ADD evals /workspace/serving/python/evals
ADD build /workspace/serving/build
ADD rcnn /workspace/serving/python/rcnn
RUN cd /workspace/serving/python/rcnn && make
RUN cd /workspace/serving/build/py_eval && make && cd -
RUN mv /workspace/serving/build/py_eval/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py_eval/inference_pb2.py /workspace/serving/python/inference_pb2.py 
RUN mv /workspace/serving/build/py_eval/inference.py /workspace/serving/python/inference.py 

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:/workspace/serving/python/rcnn:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-pulp"