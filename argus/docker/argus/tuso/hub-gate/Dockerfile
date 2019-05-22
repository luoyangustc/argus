FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./hub-gate /workspace/argus/hub-gate
ADD ./tuso-hub-gate.conf /workspace/argus/tuso-hub-gate.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./hub-gate -f ./tuso-hub-gate.conf
