# 软件授权模块
## 使用说明

### 软件以库的方式使用授权时需要

>`import _ "qiniu.com/argus/licence"`
>
>`go build -tags=ava_licence`

### 授权文件的设定
默认情况下在当前目录读取名为`ava_licence`的文件

可以通过设置环境变量`AVA_LICENCE`来修改默认路径值
>`export AVA_LICENCE=/workspace/licence.key`

### 授权校验原理
授权案需要事先采集机器信息并进行汇总记录,
然后使用gpg对采集的信息进行签名,程序运行时再根据获取的机器信息和授权文件中的进行校验

优点：
简单可靠,rsa加密保证了授权文件只有持有私钥的者才能签发

缺点：
因问公钥是嵌入到程序中的,一旦发出就不可更改,所以要绝对保证私钥的安全

### 创建gpg密钥对
1. 安装gpg,`sudo apt-get install gnupg` or `yum install gnupg`
2. 生成密钥, `gpg --gen-key`
3. 列出密钥, `gpg --list-keys`
4. 导出公钥, `gpg --armor --output public-key.txt --export [用户ID]`
5. 将导出的公钥替换到代码中

### 授权文件的格式
授权内容以json格式组织,例子如下：
```
[
    {
    "version": "v0.1",
    "expires": "2006-01-02T15:04:05+07:00",
    "os": "linux",
    "app": "seving_eval",
    "app_md5": "xxxxxxxx",
    "app_version": "v1.1",
    "data_md5": ["xxxxxx"],
    "memory_size": 10000000,
    "cpu_num": 5,
    "system_uuid": ["xxxxxxxxxx"],
    "gpu_uuid": ["xxxxxxxxxx"],
    "disk_uuid": ["xxxxxxxxxx"],
    "mac_address": ["xx:yy:zz:aa:bb"],
    },
    {
    "version": "v0.1",
    "expires": "2006-01-02T15:04:05+07:00",
    "os": "linux",
    "app": "argus_util",
    "app_md5": "xxxxxxxx",
    "app_version": "v1.1",
    "data_md5": ["xxxxxx"],
    "memory_size": 10000000,
    "cpu_num": 5,
    "system_uuid": ["xxxxxxxxxx"],
    "disk_uuid": ["xxxxxxxxxx"],
    "mac_address": ["xx:yy:zz:aa:bb"],
    }
]
```
名称 |说明|类型|匹配方式|是否必须|获取方式
----|----|---|-------|--|------
version|授权版本号|string|equal|Y|
expires|有效期|时间|less|Y|
os|操作系统|string|equal|N|
app|进程名称|string|equal|N|
app_version|程序的版本|string|equal|N|
app_md5|进程bin的md5值|string|equal|N|
data_md5|数据文件的md5值|[]string|in|N|
memory_size|内存大小|int64|less|N|
cpu_num|cpu数量|int64|less|N|
system_uuid|主板编号|[]string|in|N|`sudo dmidecode -s system-uuid`
gpu_uuid|gpu编号|[]string|in|N|`nvidia-smi --query-gpu=uuid --format=csv,noheader`
disk_uuid|硬盘uuid|[]string|in|N|`ls /dev/disk/by-uuid/ -l`
mac_address|mac地址|[]string|has|N|`ifconfig`

### 授权字段校验方式
匹配方式|意义
-------|---
equal|相应类型的值是否等于授权
less|相应的数字是否小于等于授权
in|相应的内容是否无安全包含在授权中
has|相应的内容数组中至少有一个包含在授权中

特殊约定
1. 授权文件中的必须字段(version,expires)未填写时则视为无效授权
2. 非必须字段未填写则视为忽略该项
3. 日期类型采用`RFC3339`格式表示,如：`2006-01-02T15:04:05+07:00`
4. v1版本中的一些字段没有实现,详细情况见代码

### 签名授权文件
1. 将未签名的授权文件内容保存到文件中, 如`licence.txt`
2. 签名授权文件, `gpg --clearsign licence.txt`,生成文件`licence.txt.asc`
3. 验证签名文件, `gpg --verify licence.txt.asc`


## TODO
1. 添加授权服务器支持
2. 添加多级证书支持, 允许二级证书进行授权签发
3. 授权签发工具
