FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/serving

ADD ./serving-fop /workspace/serving/serving-fop
ADD ./serving-fop.conf /workspace/serving/serving-fop.conf

RUN mkdir -p /workspace/serving/run/auditlog
ENV PATH=$PATH:/workspace/serving

WORKDIR /workspace/serving

CMD ./serving-fop -f ./serving-fop.conf
