FROM reg.qiniu.com/avaprd/ava-caffe-deluxe:201807261528-base

RUN mkdir -p /workspace/serving/run/auditlog 
RUN mkdir -p /tmp/eval

RUN pip install -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com prometheus_client

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

ADD evals /workspace/serving/python/evals
RUN g++ -fPIC /workspace/serving/python/evals/utils/caffe_log.cpp -o /workspace/serving/caffelog.so -shared -I/usr/include/python2.7 -I/usr/lib/python2.7/config-x86_64-linux-gnu/ -lglog


#Runtime set ENV OPENBLAS_NUM_THREADS=NUM_CPUS() to use multi cpu cores for caffe
ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-caffe-facex-feature"

CMD ["serving-eval","-f","serving-eval.conf"]
