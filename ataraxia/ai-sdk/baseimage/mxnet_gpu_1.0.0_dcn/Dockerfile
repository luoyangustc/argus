FROM reg.qiniu.com/avaprd/aisdk-mxnet-gpu.1.0.0:20180212-v63-dev


RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

# dcn
ENV DCN_ROOT=/opt/dcn
RUN cd opt && wget http://devtools.dl.atlab.ai/aisdk/github_branch/ataraxialab/Deformable-ConvNets/branch/dev-terror/Deformable-ConvNets2.tar.gz && tar -zxvf Deformable-ConvNets2.tar.gz && mv Deformable-ConvNets dcn
RUN cd $DCN_ROOT && sh init.sh && \
    cp -r ${DCN_ROOT}/rfcn/operator_cxx/* ${MXNET_ROOT}/src/operator/contrib/
RUN cd ${MXNET_ROOT} && \
    make clean && \
    make -j $(nproc) USE_OPENCV=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

RUN pip install -i https://mirrors.aliyun.com/pypi/simple protobuf pyzmq gevent opencv-python-headless==4.0.0.21 && python -c 'import cv2;assert cv2.__version__[0]=="4"'
