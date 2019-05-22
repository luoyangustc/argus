FROM ava-ubuntu-v1604:20170728-v26-cluster-dockfile-update

RUN mkdir -p /workspace/serving
RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /workspace/serving/data
RUN mkdir -p /tmp/eval

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=native

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-image"

CMD ["serving-eval","-f","serving-eval.conf"]

