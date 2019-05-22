---
title: 智能鉴黄服务 (pulp)
layout: document
search: true
breadcrumb: [开发者中心@/,帮助文档@/article/index.html,数据处理@/article/index.html#dora,第三方数据处理@/article/index.html#market-application]
---

智能服务`pulp`帮您智能判断在七牛云的图片是属于色情、性感还是正常。
本服务为您提供非常方便的色情检测，根据该服务提供商的评测结果显示，鉴别的准确率超过99.6%，可以替代85%以上的人工审核，并且通过本服务机器正在不断学习提高鉴别的准确率。

本服务由上海阿塔网络科技有限公司（以下简称阿塔科技）提供。
启用服务后，您存储在七牛云空间的文件将被提供给阿塔科技以供其计算使用。
七牛不能保证鉴别结果的正确性，请您自行评估后选择是否启用。
服务价格请您参考具体的价格表及计费举例，您使用本服务产生的费用由七牛代收。
启用服务则表示您知晓并同意以上内容。

<a id="pulp-open"></a>
# 如何开启

进入[七牛开发者平台](https://portal.qiniu.com/create)的 **第三方数据处理**，找到 **智能鉴黄** 点击并开始使用。

# 请求语法

```
GET <DownloadURI>?pulp HTTP/1.1
Host: <DownloadHost>
```

<a id="pulp-request-header"></a>
# 请求头部

|头部名称       | 必填 | 说明|
|:------------- | :--- | :------------------------------------------|
|Host           | 是   | 下载服务器域名，可为七牛三级域名或自定义二级域名，参考[七牛自定义域名绑定流程][cnameBindingHref]。|

# 响应语法

```
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: no-store

{
    // ...鉴黄返回值...
}
```

<a id="pulp-response-header"></a>
# 响应头部

头部名称       |  说明
:------------- | :------------------------------------------
Content-Type   | MIME类型，固定为`application/json`。
Cache-Control  | 缓存控制，固定为`no-store`，不缓存。


<a id="pulp-response-content"></a>
# 响应内容

* 如果请求成功，返回包含如下内容的JSON字符串（已格式化，便于阅读）：


```
{
	"code":  0,
    "message": "success",
    "nonce": "0.21397979712321",
    "timestamp": 22989199988,
    "pulp":{
        "rate": 0.99996,
        "label": 1,
        "review": true
    }
}
```

|字段名称         | 类型 | 说明     |
|:------------    | :--- | :-----------------------------------------------|
code              |Number| 处理状态：`0`调用成功；`1`授权失败；`2`模型ID错误；`3`没有上传文件；`4`API版本号错误；`5`API版本已弃用；`6`secretId 错误；`7`任务Id错误，您的secretId不能调用该任务；`8`secretId状态异常；`9`尚未上传证书；`100`服务器错误；`101`未知错误
message           |String| 与`code`对应的状态描述信息
timestamp         |Number|当前的服务器的Unix时间戳。
nonce	          |Number|随机数。
pulp              |Object|每个元素表示每张图片的检测结果：<br>rate：介于0-1间的浮点数，表示该图像被识别为某个分类的概率值，概率越高、机器越肯定；您可以根据您的需求确定需要人工复审的界限。<br>label：介于0-2间的整数，表示该图像被机器判定为哪个分类，分别对应： `0`色情；`1`性感；`2`正常。<br>review： 是否需要人工复审该图片，鉴黄服务是否对结果确定。`true`需要`false`不需要


* 如果请求失败，请参考以上`code`和`message`字段的说明。

* 更多参数请参考[鉴黄服务接口协议](http://kb.qiniu.com/5qwcwawm "鉴黄服务接口协议")

<a id="pulp-samples"></a>
# 示例

在Web浏览器中输入以下图片地址：

```
http://78re52.com1.z0.glb.clouddn.com/resource/gogopher.jpg?pulp
```

返回结果（内容经过格式化以便阅读）：

```
{
	"code":  0,
    "message": "success",
    "nonce": "0.21397979712321",
    "timestamp": 22989199988,
    "pulp":{
        "rate": 0.99996,
        "label": 1,
        "review": true
    }
}
```

<a id="pulp-price"></a>
# 服务价格

|总调用量P         | 确定部分       | 不确定部分     |
|:---------------- | :------------: | :------------: |
| 单位：万次       | 单价（元/百次）| 单价（元/百次）|
| P < = 300        |     0.23       |    0.0625      |
| 300 < P <=1500   |     0.21       |    0.0575      |
| 1500 < P <= 3000 |     0.19       |    0.0525      |
| P > 3000         |     0.16       |    0.045       |

**注意：**

 * 确定部分：可信度高，无需review(返回数据中review为false)
 * 不确定部分：需要人工 review，但根据返回的参考值排序可以大大降低工作量（返回数据中review为true）。

<a id="pulp-pirce-example"></a>
# 计费示例

某公司2015年5月使用七牛图片鉴黄服务，共发起500万次鉴黄请求，
其中结果确定的次数为480万次，结果不确定的次数为20万次，则当月使七牛鉴黄服务产生的费用为：

确定的结果产生费用：0.23元/百次 * 300万次 + 0.21元/百次 * (480万次 - 300万次) = 6900元 + 3780元 = 10680元

不确定的结果产生费用：0.0625元/百次 * 20万次 = 125元

总计费用：10680元 + 125元 = 10805元

[resourceHref]:  /article/newbie-guide.html#resource  "资源"
[bucketHref]:  /article/newbie-guide.html#bucket   "空间"
[keyHref]: /article/newbie-guide.html#key-value  "键值对"
[pipeHref]: /article/developer/process-mechanism.html#pipeline   "管道"
[putpolicyHref]: /article/developer/security/put-policy.html    "上传策略"
[urlsafeBase64Href]:  /article/kodo/kodo-developer/appendix.html#urlsafe-base64  "URL安全的Base64编码"
[uploadtokenHref]: /article/developer/security/upload-token.html "上传凭证"
[magicVariablesHref]: /article/kodo/kodo-developer/up/vars.html#magicvar  "魔法变量"
[xVariablesHref]: /article/kodo/kodo-developer/up/vars.html#xvar  "自定义变量"
[callbackHref]: /article/kodo/kodo-developer/up/response-types.html#callback   "回调通知"
[returnbodyHref]: /article/developer/responsebody.html#response-body "自定义响应内容"
[mkblkHref]:  /code/v6/api/kodo-api/up/mkblk.html   "创建块"
[bputHref]:  /code/v6/api/kodo-api/up/bput.html   "上传片"
[mkfileHref]: /code/v6/api/kodo-api/up/mkfile.html  "创建文件"
[varsHref]:  /article/kodo/kodo-developer/up/vars.html "变量"
[httpcodeHref]: /article/developer/response-body.html#http-code  "HTTP状态码"
[httpextenderHref]: /article/developer/response-body.html#http-extender  "HTTP扩展字段"
[urlescapeHref]: /article/glossary/#url-encoding "URL编码"
[avfopHref]: /article/developer/persistent-fop.html "持久化数据处理"
[simpleHref]: /article/kodo/kodo-developer/up/response-types.html#simple-response  "简单反馈"
[rsopHref]: /article/index.html#kodo-api-handbook#rs  "资源管理"
[fopHref]: /article/index.html#dora-api-handbook  "多媒体数据处理API参考手册"
[asynchronousHref]:  /article/kodo/kodo-developer/up/response-types.html#persistent-op  "异步数据处理"
[accesstokenHref]: /article/developer/security/access-token.html   "管理凭证"
[imagefopHref]: /code/v6/api/kodo-api/image/index.html "图片处理"
[pfopHref]: /article/developer/persistent-fop.html#pfop-existing-resource "对已有资源手动触发"
[persistentopsHref]: /article/developer/security/put-policy.html#put-policy-persistent-ops-explanation  "persistentOps详解"
[principleHref]: /article/kodo/kodo-developer/download-process.html#download-mechanism    "下载机制"
[qrsctlHref]:  /code/v6/tool/qrsctl.html "qrsctl"
[encodedEntryURIHref]: /article/developer/format.html#encodentry "EncodedEntryURI格式"
[hmacsha1Href]: /article/glossary/#h  "HMAC-SHA1"
[sendBugReportHref]:   mailto:support@qiniu.com?subject=599错误日志     "发送错误报告"
[mpsHref]: https://portal.qiniu.com/create/mps  "专用队列"
[prefopHref]: /code/v6/api/dora-api/pfop/prefop.html  "持久化处理状态查询"
[cnameBindingHref]:  http://kb.qiniu.com/53a48154  "七牛自定义域名的绑定流程"
[exifHref]: /code/v6/api/kodo-api/image/exif.html  "EXIF"
[persistentOpsHref]: /article/developer/security/put-policy.html#put-policy-persistent-ops  "预转持久化处理"
[saveasHref]: /code/v6/api/dora-api/saveas.html  "处理结果另存"
[imageMogr2Href]: /code/v6/api/kodo-api/image/imagemogr2.html   "图片高级处理"
[ExifWhitePaperHref]:  http://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf   "Exif技术白皮书"
[resourceProtectHref]:  http://kb.qiniu.com/52uad43y  "原图保护"

[videowatermarkHref]:  /code/v6/api/dora-api/av/video-watermark.html  "视频水印"
[thumbnailHref]:                ../../list/thumbnail.html                       "缩略图文档列表"
[avthumbHref]: /code/v6/api/dora-api/av/video-watermark.html  "音视频转码"
[watermarkHref]: /code/v6/api/kodo-api/image/watermark.html  "图片水印处理"
[download-tokenHref]: /article/developer/security/download-token.html   "下载凭证"
[securityHref]: /article/developer/security/index.html "安全机制"
[portalHref]:  https://portal.qiniu.com "七牛开发者平台"
[unixTimeHref]:  /article/glossary/#u "unix时间戳"
[responsetypesHref]: /article/kodo/kodo-developer/up/response-types.html    "响应类型"
[jsonHref]:                 /article/glossary/#j        "JSON格式"
[listHref]: /code/v6/api/kodo-api/rs/list.html    "列举资源"
[ufopfastHref]:  /article/dora/ufop/ufop-fast.html "自定义数据处理快速上手"
