FROM ubuntu:16.04
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates wget curl vim git unzip python-dev iputils-ping net-tools netcat telnet htop lsof && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget http://devtools.dl.atlab.ai/docker/PRC-tz -O /tmp/PRC-tz && mv /tmp/PRC-tz /etc/localtime
RUN wget -O /usr/local/bin/dumb-init http://devtools.dl.atlab.ai/docker/dumb-init_1.2.0_amd64 && \
    chmod +x /usr/local/bin/dumb-init
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
