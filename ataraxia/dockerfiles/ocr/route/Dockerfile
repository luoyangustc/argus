FROM reg.qiniu.com/inference/terror-det-sup-cls-refinedet:20180713-v1

RUN mkdir -p /workspace/serving/python/evals

RUN pip install prometheus_client

ADD build /workspace/serving/build
ADD evals /workspace/serving/python/evals
RUN cd /workspace/serving/build/py && make && cd -
RUN mv /workspace/serving/build/py/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py/inference_pb2.py /workspace/serving/python/inference_pb2.py
RUN mv /workspace/serving/build/py/inference_py.py /workspace/serving/python/inference_py.py


ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-card-detect-refinedet"