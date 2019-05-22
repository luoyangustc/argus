FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./hub-process /workspace/argus/hub-process
ADD ./tuso-hub-process.conf /workspace/argus/tuso-hub-process.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./hub-process -f ./tuso-hub-process.conf
