FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./argus-job-gate /workspace/argus/argus-job-gate
ADD ./argus-job-gate.conf /workspace/argus/argus-job-gate.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-job-gate -f ./argus-job-gate.conf
