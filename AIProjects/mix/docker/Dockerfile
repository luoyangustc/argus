FROM BASEIMAGE


RUN apt-get update -y \
    && apt-get -y install python-setuptools python-pip libpython2.7-dev curl wget
RUN mkdir -p /usr/share/zoneinfo/Asia
RUN wget http://pj7bc1tg0.bkt.clouddn.com/Shanghai -O /usr/share/zoneinfo/Asia/Shanghai
RUN rm -f /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN python -m pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com prometheus_client 

RUN mkdir -p /workspace/serving/run/auditlog && \
    mkdir -p /tmp/eval

ADD mix /workspace/serving/mix
ADD mix.conf /workspace/serving/mix.conf
ADD run.cmd /workspace/serving/run.cmd

WORKDIR /workspace/serving

ENTRYPOINT [ "/bin/bash", "./run.cmd" ]
