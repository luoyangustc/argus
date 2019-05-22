# TODO replace by image in cs hub
FROM reg.qiniu.com/argusprd/ava-mxnet-gpu.1.0.0:20180502-v64-base

RUN mkdir -p /workspace/serving
RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /tmp/eval

RUN pip install prometheus_client

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

ADD evals /workspace/serving/python/evals

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=.:/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-mxnet-classify-v3"

CMD ["serving-eval","-f","serving-eval.conf"]
