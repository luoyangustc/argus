FROM reg.qiniu.com/inference/ataraxia-ocr-weixin-weibo-text-recognition:20180202-v56-dev

RUN mkdir -p /workspace/serving/run/auditlog 
RUN mkdir -p /tmp/eval

RUN pip install prometheus_client

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

#Runtime set ENV OPENBLAS_NUM_THREADS=NUM_CPUS() to use multi cpu cores for caffe
ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=lib

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-caffe-ocr-recognition"

CMD ["serving-eval","-f","serving-eval.conf"] 