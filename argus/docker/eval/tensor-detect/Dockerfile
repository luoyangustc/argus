FROM tensorrt

RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /workspace/serving/run/tensorrt
RUN mkdir -p /tmp/eval

ADD serving-eval.conf /workspace/serving/serving-eval.conf
ADD libtensorrt.so /workspace/serving/libtensorrt.so
ADD serving-eval /workspace/serving/serving-eval
RUN chmod +x /workspace/serving/serving-eval
ENV PATH=$PATH:/workspace/serving
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/serving
ENV TENSOR_TYPE=detect

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-tensor-detect"

CMD ["serving-eval","-f","serving-eval.conf"]
