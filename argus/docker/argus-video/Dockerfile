FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./argus-video /workspace/argus/argus-video
ADD ./argus-video.conf /workspace/argus/argus-video.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-video -f ./argus-video.conf
