FROM ava-caffe-deluxe:20170606-v32-dev

RUN apt-get update && apt-get install libsasl2-dev && apt-get install axel
RUN axel -n 10 https://storage.googleapis.com/golang/go1.8.3.linux-amd64.tar.gz -o go1.8.3.tar.gz
RUN tar xzvf go1.8.3.tar.gz

RUN mkdir -p /workspace/serving/code
ADD platform /workspace/serving/code/platform

ENV GOROOT=/workspace/go
ENV GOPATH=/workspace/serving/code/platform
RUN cd /workspace/serving/code/platform && /workspace/go/bin/go install ./...
RUN cp /workspace/serving/code/platform/bin/hack-embed-eval /workspace/serving/hack-embed-eval 

RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /tmp/eval

ADD evals /workspace/serving/python/evals
RUN g++ -fPIC /workspace/serving/python/evals/utils/caffe_log.cpp -o /workspace/serving/caffelog.so -shared -I/usr/include/python2.7 -I/usr/lib/python2.7/config-x86_64-linux-gnu/ -lglog

# example
ADD example.sh /workspace/serving/example.sh
ADD model.tar /tmp/eval/init/
ADD example.files /workspace/serving/example.files

#Runtime set ENV OPENBLAS_NUM_THREADS=NUM_CPUS() to use multi cpu cores for caffe
ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH
ENV USE_DEVICE=CPU

WORKDIR /workspace/serving
