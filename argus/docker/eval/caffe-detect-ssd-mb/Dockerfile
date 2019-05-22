# TODO replace by image in cs hub
FROM ava-caffe-facex-detect:ssd-mobilenet

RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /tmp/eval

RUN pip install prometheus_client

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

ADD evals /workspace/serving/python/evals
ADD build /workspace/serving/build
RUN cd /workspace/serving/build/py_eval && make && cd -
RUN mv /workspace/serving/build/py_eval/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py_eval/inference_pb2.py /workspace/serving/python/inference_pb2.py 
RUN mv /workspace/serving/build/py_eval/inference.py /workspace/serving/python/inference.py 

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH
ENV INTEGRATE=lib

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-caffe-detect-ssd-mb"

CMD ["serving-eval","-f","serving-eval.conf"]