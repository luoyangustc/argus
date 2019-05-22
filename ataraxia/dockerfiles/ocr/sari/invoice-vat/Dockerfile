FROM reg-xs.qiniu.io/atlab/ctpn:20180118-torch0.3.0

RUN mkdir -p /workspace/serving/python/evals

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN pip install prometheus_client
RUN pip install scikit-learn -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

ADD build /workspace/serving/build
ADD evals /workspace/serving/python/evals
ADD src /workspace/serving/python/evals/src

RUN cd /workspace/serving/build/py && make && cd -
RUN mv /workspace/serving/build/py/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py/inference_pb2.py /workspace/serving/python/inference_pb2.py 
RUN mv /workspace/serving/build/py/inference_py.py /workspace/serving/python/inference_py.py 

ENV PYTHONPATH=/workspace/serving/python/evals/src:/workspace/serving/python:$PYTHONPATH
WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-ocr-sari-vat"