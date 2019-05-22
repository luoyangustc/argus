FROM ava-caffe-deluxe:20170606-v32-dev

RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /tmp/eval

RUN pip install docopt
ADD evals /workspace/serving/python/evals
RUN g++ -fPIC /workspace/serving/python/evals/utils/caffe_log.cpp -o /workspace/serving/caffelog.so -shared -I/usr/include/python2.7 -I/usr/lib/python2.7/config-x86_64-linux-gnu/ -lglog

# example
ADD example.sh /workspace/serving/example.sh
ADD model.tar /tmp/eval/init/
ADD example.files /workspace/serving/example.files

ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
