FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./ccp-manual /workspace/argus/ccp-manual
ADD ./ccp-manual.conf /workspace/argus/ccp-manual.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./ccp-manual -f ./ccp-manual.conf
