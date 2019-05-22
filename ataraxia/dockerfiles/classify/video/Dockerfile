# TODO replace by image in cs hub
FROM ava-caffe-deluxe.vedio.classify.senet:20180110-v59-vedio_classfiy

RUN apt-get install  -y --no-install-recommends  libprotobuf-dev libsnappy-dev libopenblas-dev protobuf-compiler && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean
    
RUN mkdir -p /workspace/serving

RUN pip install prometheus_client && pip install easydict && pip install protobuf

ADD evals /workspace/serving/python/evals
ADD build /workspace/serving/build
RUN cd /workspace/serving/build/py_eval && make && cd -
RUN mv /workspace/serving/build/py_eval/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py_eval/inference_pb2.py /workspace/serving/python/inference_pb2.py 
RUN mv /workspace/serving/build/py_eval/inference.py /workspace/serving/python/inference.py 

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-video-classify"