FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./argus-gate /workspace/argus/argus-gate
ADD ./argus-gate.conf /workspace/argus/argus-gate.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-gate -f ./argus-gate.conf
