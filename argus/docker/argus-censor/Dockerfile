FROM reg.qiniu.com/avaprd/argus-base/serving-base:20180713-v63-dev

RUN mkdir -p /workspace/argus

ADD ./argus-censor /workspace/argus/argus-censor
ADD ./argus-censor.conf.json /workspace/argus/argus-censor.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-censor -f ./argus-censor.conf
