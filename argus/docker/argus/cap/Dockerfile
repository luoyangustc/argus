FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./argus-cap /workspace/argus/argus-cap
ADD ./argus-cap.conf /workspace/argus/argus-cap.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-cap -f ./argus-cap.conf