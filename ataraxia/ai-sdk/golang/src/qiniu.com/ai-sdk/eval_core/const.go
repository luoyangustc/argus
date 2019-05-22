package main

const INFERENCE_ZMQ_ADDR = "tcp://127.0.0.1:9655" // 对外暴露的推理API zmq协议
const INFERENCE_GRPC_ADDR = ":9009"               // 对外暴露的推理API grpc协议
// const INFERENCE_HTTP_ADDR = ":9100"               // 对外暴露的推理API http协议 (由serving-eval提供)
const MONITOR_HTTP_ADDR = ":9101" // 对外暴露的推理API zmq协议

const INFERENCE_ZMQ_IN = "tcp://127.0.0.1:9654" // eval_core bind，inference connect， eval_core 发送请求给 inference 使用的套接字

const MONIROT_ZMQ_ADDR = "tcp://127.0.0.1:9401" // eval_core 监听的收集监控信息的zmq套接字

const INFERENCE_FORWARD_IN = "tcp://127.0.0.1:9301" // inference 连接 eval_core 发送推理请求

const FORWARD_IN = "tcp://127.0.0.1:9201"
const FORWARD_OUT = "tcp://127.0.0.1:9200"
