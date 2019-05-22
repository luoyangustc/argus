# TODO replace by image in cs hub
FROM index.qiniu.com/ataraxia/ava-mxnet:gpu

# dcn
ENV DCN_ROOT=/opt/dcn
RUN git clone -b dev-terror https://github.com/ataraxialab/Deformable-ConvNets.git ${DCN_ROOT} && \
    cd $DCN_ROOT && sh init.sh && \
    cp -r ${DCN_ROOT}/rfcn/operator_cxx/* ${MXNET_ROOT}/src/operator/contrib/
RUN cd ${MXNET_ROOT} && \
    make clean && \
    make -j $(nproc) USE_OPENCV=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

RUN mkdir -p /workspace/serving
RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /tmp/eval

RUN pip install prometheus_client pyyaml

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf
ADD resnet.yaml /workspace/serving/resnet.yaml

ADD evals /workspace/serving/python/evals

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=.:/workspace/serving/python:$DCN_ROOT:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-mxnet-detect-dcn"

CMD ["serving-eval","-f","serving-eval.conf"]
