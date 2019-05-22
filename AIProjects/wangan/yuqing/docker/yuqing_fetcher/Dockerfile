FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev
RUN mkdir -p /workspace/argus/run/auditlog
RUN wget https://github.com/lionsoul2014/ip2region/raw/master/data/ip2region.db -O /workspace/argus/ip2region.db
ADD ./yuqing_fetcher /workspace/argus/yuqing_fetcher
ADD ./yuqing_fetcher.conf /workspace/argus/yuqing_fetcher.conf
ENV PATH=$PATH:/workspace/argus
EXPOSE 9000
WORKDIR /workspace/argus
CMD ./yuqing_fetcher -f yuqing_fetcher.conf