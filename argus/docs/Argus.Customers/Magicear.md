
# API

## /v1/video/<id>

> 视频检测奖杯

*Request*

```
POST /v1/video/<id> HTTP/1.1
Host: argus.atlab.ai
Content-Type: application/json
Authorization: Qiniu xxxxx:xxxxxxx
Content-Length: 2222

{
    "data": {
        "uri": "http://host/path"
    },
    "params": {
        "async": true,
        "vframe": {
            "interval": 2
        }
    },
    "ops": [
        {
            "op": "magicear-trophy",
            "hookURL": "http://host/path"
        }
    ]
}
```

* **id**: `string` 标识视频，callback时会带上该唯一标识
* **data.uri**: `string` 视频地址，支持mp4、flv
* **data.params.async**: `bool` 是否启用异步。如果视频内容较长（比如5分钟以上），建议启用异步，结果会通过callback形式返回；否则会超时，超时时间为30s。
* **data.vframe.interval**: `float` 视频截帧时间间隔，单位是秒。
* **ops.op**: `string` 具体执行的操作，这个案例固定为`magicear-trophy`
* **ops.hookURL**: `string` callback地址，不填则不进行callback。


*Response*

```
HTTP/1.1 200 OK
```

*Callback*

```
POST / HTTP/1.1
Host: host
Content-Type: application/json
Content-Length: 2222

{
	"id": <id>,
	"result": {
		"segments": [
			{
				"op":"",
				"offset_begin": 450000,
				"offset_end": 458000,
				"label": "trophy",
				"score": 0.9995,
				"cuts": [
					{
						"offset":454000,
						"result": {
							"code": 0,
							"message": "",
							"result": {
								"detections": [
									{
										"index":1,
										"class": "trophy",
										"score":0.9994,
										"pts":[[93,251],[228,251],[228,364],[93,364]]
									},
									{
										"index": 2,
										"class":"message",
										"score":0.9961,
										"pts":[[1012,425],[1212,425],[1212,457],[1012,457]]
									}
								]
							}
						}
					}
				]
			},
			{
				"op":"",
				"offset_begin":570000,
				"offset_end":588000,
				"label":"trophy",
				"score":0.9992,
				"cuts":[]
			}
		]
	}
}
```