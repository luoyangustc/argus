# 城运接口文档

## 1. 抓拍数据库接口

### 1.1。 元数据数据库
> 目标数据库为oracle

**Request**
```
POST /v1/fetch_imgs

{
	"camera_id":"NHPZXPZ001T1",
	"camera_ip":"pdcp04/192_168_72_155",
	"start_time":1521686990,
	"duration":50    
}
```

|名称|类型|说明|
|:---|:---|:---|
|camera_id|string|抓拍相机ID|
|camera_ip|string|抓拍相机地址信息，用于获取抓拍图片|
|start_time|Unix时间戳|查询起始时间戳|
|duration|int|持续时间，秒为单位|


**Response**
{
    "base_url": "/static/pdcp04/192_168_72_155/",
    "imgs": [
        {
            "CDBH": 1,
            "CLSD": "沪D",
            "DMDM": "NHPZXPZ001T1",
            "HPHM": "沪D05293",
            "HPYS": "01",
            "JGSJ": 1521686993,
            "JGSJ_STR": "2018-03-22T10:49:53",
            "TPID": "NHPZXPZ00000001120180322104953990",
            "file_name": "2018/03/22/10/NHPZXPZ00000001120180322104953990_沪D05293_01_Z_hptp.jpg"
        },
        ...
    ],
    "param": {
        "camera_id": "NHPZXPZ001T1",
        "camera_ip": "pdcp04/192_168_72_155",
        "duration": 50,
        "start_time": 1521686990
    }
}

|名称|Key|类型|说明|
|:---|:---|:---|:---|
|base_url|string|查询结果的基础拼接url,用于获取资源文件|
|imgs.CDBH|int|抓拍摄像机的固定车道号|
|imgs.CLSD|string|分辨是沪牌还是外地车牌|
|imgs.DMDM|string|唯一抓拍编号，会对应至多两张抓拍图，车票图和全景图|
|imgs.HPHM|string|系统识别出的车牌号，可能有未识别|
|imgs.HPYS|int|车牌颜色，分别蓝牌车和黄牌车，可能与未识别|
|imgs.JGSJ|时间戳类型|抓拍摄像头的抓拍时间|
|imgs.JGSJ_STR|string|抓拍摄像头的抓拍时间|
|imgs.TPID|string|抓拍唯一编号|
|imgs.file_name|string|抓拍图片访问路径，与base_url拼接访问资源|
|params.camera_id|string|抓拍相机ID|
|params.camera_ip|string|抓拍相机地址信息，用于获取抓拍图片|
|params.start_time|Unix时间戳|查询起始时间戳|
|params.duration|int|持续时间，秒为单位|

### 1.2. 图片访问
> smb文件服务器

```
文件名按IP+日期+时间+抓拍编号+车牌号+车牌类别+车牌/全景 来命名图片

GET http://10.118.61.240:7756/static/pdcp04/192_168_72_155/2018/03/22/10/NHPZXPZ00000001120180322104953990_沪D05293_01_Z_hptp.jpg
```

**注意**
* 按抓拍编号去获取图片

## 2. 硬盘录像机接口

### 2.1. 视频获取

**命令**

```
./DvrDL -uid pudong -start_time 20180318120000 -prefix /home/qnai/dvr_porting/

目前-uid 是预留给后续做摄像头和用户名密码映射用的，现在只有一个设想头，这个值可以任意

如果网络，设备是ok 的，会在/home/qnai/dvr_porting/ 下产生如下的文件

20180318115220_20180318120339.mp4
```


### 2.2. 视频截帧
**命令**

```
./ExtractFrames -i HK_Ori.mp4 -ss 20 -to 40 -r 1 -prefix ./

-i 后跟输入url

-ss 后跟开始时间点(使用文件内的相对时间)

-to 后跟结束时间点(使用文件内的相对时间)

-r  表示截帧的帧率

-prefix 表示输出路径
```

### 2.3. 视频截取
**命令**

```
./ExtractVideo -i HK_Ori.mp4 -ss 20 -to 40 -o normal_1.mp4

-i 后跟输入url

-ss 后跟开始时间点(使用文件内的相对时间)

-to 后跟结束时间点(使用文件内的相对时间)

-o 表示输出路径
```

## 3. 黄牌车查证请求
**Request**
```
Post /v1/eval/zhatu
Content-Type: application/json

{
    "data": {
        "uri": "http://oqgascup5.bkt.clouddn.com/NHPZXPZ00000001320180303171650920_%E6%B2%AABL3623_01_Z_qjtp.jpg",
        "attribute": {
            "image_type": 0
        }
    }
}

```

**Response**
```
200 OK
Content-Type： application/json

{
    "code": 0,
    "message": "",
    "result": {
        "detections": [
            {
                "label": 0,
                "class": 0,
                "score": 0.9937129616737366,
                "pts": [776,170,1556,1191]
            },
            ...
        ]
    }
}
```

**请求字段说明**：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|attribute.image_type|int|图片类别|

**结果字段说明**：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|label|int|车辆分类，其中0表示渣土车，1表示其他大型车辆|
|class|int|覆盖识别情况，TODO|
|score|float|将图片判别为某一类的准确度，取值范围0~1，1为准确度最高|
|pts|两点坐标框|车辆在图片中位置，用[左上X坐标，左上Y坐标，右下X坐标，右下Y坐标]|



## 4. 渣土信息出口
### 4.1. 上传渣土车查证信息
**Request**
```
Post <Address>/v1/zhatuche/capture/<id>
Content-Type: application/json

{
    "time": <timestamp>,
    "camera_id": <camera_id>,
    "camera_info": <camera_info>,
    "licence_id": <licence_id>,
    "licence_type": <licence_type>,
    "lane": <lane>,
    "result":<result>,
    "score":<score>,
    "coordinate": {
		"gps": <gps_address>
	},
    "resource": {
        "images": [
            {
                "uri": <image_download_uri>,
                "pts": <pts>
            },            
            ...
        ],
        "videos": [
            <video_download_uri>,
            ...
        ]
    }
}

```

|名称|类型|说明|
|:---|:---|:---|
|id|string|渣土车识别唯一ID，复用抓拍的ID|
|timestamp|time|号牌抓拍入库的时间戳|
|camera_id|string|交管中心球机摄像头ID|
|camera_info|string|球机描述信息，表明卡口位置|
|licence_id|string|车牌号（含未识别|
|licence_type|int|黄牌和未识别|
|lane|int|车道|
|result|int|已覆盖、未覆盖、疑似未覆盖(0、1、2)|
|score|float|识别置信度|
|coordinate.gps|gps坐标|经纬度坐标，例如[31.202636,121.513196]|
|resource.images.uri|string|查证图片下载链接，jpeg格式|
|resource.images.pts|矩形坐标框，int数组|车辆在图片中位置，用[左上X坐标，左上Y坐标，右下X坐标，右下Y坐标]|
|resource.videos|string数组|查证视频下载链接，mp4格式|

## 5. 系统接口
### 5.1. 获取指定抓拍
**Request**
```
POST /v1/capture

{
    "camera_id":"NHPZXPZ001T1",
	"camera_ip":"pdcp04/192_168_72_155",
	"start_time":"20180323120000",
	"duration":300
}
```

|名称|类型|说明|
|:---|:---|:---|
|camera_id|string|抓拍相机ID|
|camera_ip|string|抓拍相机地址信息，用于获取抓拍图片|
|start_time|string|查询起始时间，北京时间|
|duration|int|持续时间，秒为单位|

**Response**
```
200 OK
```

### 5.2. 缓存监控录像
**Request**
```
POST /v1/video/time
{
    "time":"20180322151800"
}
```

|名称|类型|说明|
|:---|:---|:---|
|time|string|监控视频查询点，北京时间|

**Response**
```
200 OK
```

### 5.3. 生成渣土车包
**Request**
```
POST /v1/capture/archive

{
	"start_time":"20180323120000",
	"end_time":"20180323120000",
	"illegal": false
}
```

|名称|类型|说明|
|:---|:---|:---|
|start_time|string|查询起始时间，北京时间|
|end_time|string|查询起始时间，北京时间|
|illegal|bool|是否只传违规渣土车，北京时间|

**Response**
```
200 OK
```
