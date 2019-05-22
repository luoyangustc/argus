FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./monitor /workspace/argus/monitor

ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./monitor
