FROM reg.qiniu.com/inference/ava-mxnet-gpu.1.0.0:20180212-v63-dev


RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

# dcn
ENV DCN_ROOT=/opt/dcn
RUN git clone -b dev-terror https://github.com/ataraxialab/Deformable-ConvNets.git ${DCN_ROOT} && \
    cd $DCN_ROOT && sh init.sh && \
    cp -r ${DCN_ROOT}/rfcn/operator_cxx/* ${MXNET_ROOT}/src/operator/contrib/
RUN cd ${MXNET_ROOT} && \
    make clean && \
    make -j $(nproc) USE_OPENCV=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

RUN mkdir -p /workspace/serving/python/evals
#RUN mkdir -p /workspace/serving/run/auditlog
#RUN mkdir -p /tmp/eval

RUN pip install prometheus_client pyyaml
RUN pip --default-timeout=100 install -U protobuf
RUN pip install easydict 
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        build-essential \
#        libprotobuf-dev \
#        protobuf-compiler
#RUN apt-cache search protobuf
RUN apt-get update && apt-get install  -y --no-install-recommends  libprotobuf-dev libsnappy-dev libopenblas-dev protobuf-compiler && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean


ADD resnet.yaml /workspace/serving/python/evals/resnet.yaml

ADD build /workspace/serving/build
ADD evals /workspace/serving/python/evals
RUN cd /workspace/serving/build/py && make && cd -
RUN mv /workspace/serving/build/py/inference.so /workspace/serving/inference.so
RUN mv /workspace/serving/build/py/inference_pb2.py /workspace/serving/python/inference_pb2.py
RUN mv /workspace/serving/build/py/inference_py.py /workspace/serving/python/inference_py.py

#ADD serving-eval /workspace/serving/serving-eval
#ADD serving-eval.conf /workspace/serving/serving-eval.conf

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$DCN_ROOT:$PYTHONPATH
RUN echo $PYTHONPATH
#terror-detect
WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-mxnet-detect-dcn"

#CMD ["serving-eval","-f","serving-eval.conf"]
