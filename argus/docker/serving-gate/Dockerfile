FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/serving

ADD ./serving-gate /workspace/serving/serving-gate
ADD ./serving-gate.conf /workspace/serving/serving-gate.conf

RUN mkdir -p /workspace/serving/run/auditlog
ENV PATH=$PATH:/workspace/serving

WORKDIR /workspace/serving

CMD ./serving-gate -f ./serving-gate.conf
