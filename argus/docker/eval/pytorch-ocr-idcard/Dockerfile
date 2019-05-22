FROM  ataraxia-inference-ocr-idcard-pytorch:20180126-v28-new-deploy-ocr-idcard

RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /tmp/eval

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

ENV INTEGRATE=lib

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-pytorch-ocr-idcard"

CMD ["serving-eval","-f","serving-eval.conf"]
