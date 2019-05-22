# TODO replace by image in cs hub
FROM index.qiniu.com/ataraxia/ava-mxnet:gpu

RUN mkdir -p /workspace/serving
RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /tmp/eval

RUN pip install prometheus_client

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

ADD evals /workspace/serving/python/evals
ADD rcnn /workspace/serving/python/rcnn
RUN cd /workspace/serving/python/rcnn && make
RUN pip install pycrayon

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=.:/workspace/serving/python:/workspace/serving/python/rcnn:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-mxnet-detect"

CMD ["serving-eval","-f","serving-eval.conf"]
