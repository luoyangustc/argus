FROM ubuntu:16.04
RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-dev
ADD pip.conf /root/.pip/pip.conf
RUN pip3 install --upgrade pip sqlalchemy==1.2.5 cx-Oracle==6.2.1
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y wget unzip libaio1 cifs-utils
RUN mkdir -p /opt/oracle && cd /opt/oracle && wget http://q.hi-hi.cn/instantclient-basic-linux.x64-12.2.0.1.0.zip && unzip instantclient-basic-linux.x64-12.2.0.1.0.zip && sh -c "echo /opt/oracle/instantclient_12_2 > /etc/ld.so.conf.d/oracle-instantclient.conf" && ldconfig
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_12_2:$LD_LIBRARY_PATH
RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
WORKDIR /src
RUN pip3 install --upgrade schedule pytest jsonschema flask schema jupyter flask_restful requests gunicorn
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y vim
CMD /src/init.sh
ADD * /src/
