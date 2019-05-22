FROM reg.qiniu.com/avaprd/aisdk-mxnet_gpu_1.0.0_dcn:20190214-v31-dev
RUN pip install -i https://mirrors.aliyun.com/pypi/simple prometheus_client protobuf pyzmq schema gevent
RUN mkdir -p /workspace/serving
RUN mkdir -p /tmp/eval
WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-{{app_name}}"
CMD ["/src/res_build/eval_core", "daemon", "-f", "/src/python/aisdk/app/{{app_name}}/eval-core.conf"]
ENV PYTHONPATH=/opt/mxnet/python:/opt/dcn:/opt/dcn/rfcn:/opt/dcn/lib:/src/python

#{% if copy_src %}
#部分文件来源于terror_detect
ADD {{cfg.build.model_tar[:-4].replace('res', 'res_build')}}/ /src/{{cfg.build.model_tar[:-4]}}/
ADD res_build/serving-eval /src/res_build/serving-eval
ADD python/aisdk/app/terror_detect/serving-eval.conf /src/python/aisdk/app/{{app_name}}/serving-eval.conf
ADD res_build/eval_core /src/res_build/eval_core
ADD python/aisdk/app/terror_detect/eval-core.conf /src/python/aisdk/app/{{app_name}}/eval-core.conf

ADD python/aisdk/proto /src/python/aisdk/proto
ADD python/aisdk/common /src/python/aisdk/common
ADD python/aisdk/framework /src/python/aisdk/framework
ADD python/aisdk/__init__.py /src/python/aisdk/__init__.py
ADD python/aisdk/app/__init__.py /src/python/aisdk/app/__init__.py
ADD python/aisdk/app/{{app_name}} /src/python/aisdk/app/{{app_name}}
ADD python/aisdk/app/terror_detect/forward.py python/aisdk/app/terror_detect/inference.py python/aisdk/app/terror_detect/resnet.yaml /src/python/aisdk/app/{{app_name}}/
#{% endif %}
