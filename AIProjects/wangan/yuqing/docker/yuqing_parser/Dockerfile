FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev
RUN mkdir -p /workspace/argus/run/auditlog
ADD ./yuqing_parser /workspace/argus/yuqing_parser
ADD ./yuqing_parser.conf /workspace/argus/yuqing_parser.conf
ADD ./result.html /workspace/argus/result.html
ENV PATH=$PATH:/workspace/argus
EXPOSE 9000
WORKDIR /workspace/argus
CMD ./yuqing_parser -f yuqing_parser.conf