/*
image_sync 包描述了同步图片请求的场景下具备的组件集合

监控组件

监控项列表

    API 服务请求数
    API 服务响应数 (区分正常 / 异常响应)
    API 平均响应时间
    API 95峰值响应时间
    API 服务请求成功率
    子服务请求数
    子服务响应数 (区分正常 / 异常响应)
    子平均响应时间
    子服务95峰值响应时间
    子服务请求成功率

监控指标列表

    metric 名称                                             类型            labels 列表
    qiniu_ai_image_argus_service_request_count             counter         service
    qiniu_ai_image_argus_service_response_time             histogram       service, code
    qiniu_ai_image_argus_sub_service_request_count         counter         sub_service, service
    qiniu_ai_image_argus_sub_service_response_time         histogram       sub_service, service, code

注：

    service: 服务名称
    code：请求返回码，无特殊说明，默认指 HTTP code
    sub_service: 子服务名称，可以为调用的底层原子服务，也可以是其他通过服务（例如拉取资源、图片处理等）
    argus_service_request_resouce_size 的 Bucket 分布为：`1e3, 1e4, 1e5, 1e6, 2 * 1e6, 4 * 1e6, 8 * 1e6, 16 * 1e6, 32 * 1e6, 64 * 1e6`
    qiniu_ai_image_argus_service_response_time 和 qiniu_ai_image_argus_sub_service_response_time 的 Bucket 分布为：`0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60`

*/

package image_sync
