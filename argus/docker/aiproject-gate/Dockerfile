FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./aiproject-gate /workspace/argus/aiproject
ADD ./aiproject.conf /workspace/argus/aiproject.conf
ADD ./ca.pem /workspace/argus/ca.pem
ADD ./ca.key /workspace/argus/ca.key

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./aiproject -f ./aiproject.conf
