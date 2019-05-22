FROM reg.qiniu.com/avaprd/argus-base-video-base:20180906-v693-dev

RUN mkdir -p /workspace/argus

RUN mkdir -p /workspace/argus/run/auditlog

ADD ./argus-vframe /workspace/argus/argus-vframe
ADD ./argus-vframe.conf /workspace/argus/argus-vframe.conf

ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-vframe -f ./argus-vframe.conf
