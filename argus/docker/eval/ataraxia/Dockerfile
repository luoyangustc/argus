FROM BASEIMAGE

RUN apt-get update -y \
    && apt-get -y install python-setuptools python-pip libpython2.7-dev \
    && pip install -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com prometheus_client \
    && mkdir -p /workspace/serving/run/auditlog \
    && mkdir -p /tmp/eval

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

#Runtime set ENV OPENBLAS_NUM_THREADS=NUM_CPUS() to use multi cpu cores for caffe
ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=lib

WORKDIR /workspace/serving

CMD ["serving-eval","-f","serving-eval.conf"]
