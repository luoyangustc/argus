FROM reg.qiniu.com/avaprd/argus-base/serving-base:20180713-v63-dev

RUN mkdir -p /workspace/argus

ADD ./argus-fop /workspace/argus/argus-fop
ADD ./argus-fop.conf /workspace/argus/argus-fop.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-fop -f ./argus-fop.conf
