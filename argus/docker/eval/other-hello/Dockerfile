FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/serving
RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /workspace/serving/data

ADD ./serving-eval /workspace/serving/serving-eval
ADD ./serving-eval.conf /workspace/serving/serving-eval.conf

ENV USE_ENV=true
ENV PATH=$PATH:/workspace/serving

WORKDIR /workspace/serving

CMD ["serving-eval","-f","serving-eval.conf"]
