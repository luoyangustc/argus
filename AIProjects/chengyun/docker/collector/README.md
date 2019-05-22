# 文档整理


http://cx-oracle.readthedocs.io/en/latest/installation.html#overview
http://docs.sqlalchemy.org/en/latest/orm/tutorial.html
https://cf.qiniu.io/pages/viewpage.action?pageId=67716234   城运中心 - 系统设计 - 技术调研


## 技术相关

挂载cifs

```bash
mkdir -p /mnt/samba && mount -t cifs //10.118.84.19/opt /mnt/samba -o username=xx,password=,iocharset=utf8
```

卸载

```bash
umount /mnt/samba
```


## API

```bash
curl -v "http://10.118.61.240:7756/v1/fetch_imgs" -X "POST" -H "Content-Type: application/json" -d '{"camera_id":"A20PVRTRAN030", "camera_ip":"pdcp03/192_168_22_104", "start_time":1521521182, "duration":10}'
```

## 已知问题

- 生成的url很小概率404，主要出现在59分钟处
- 图片文件只有最近一个月的
- 单元测试跑不过，疑似某个摄像头挂了几小时

## TODO

- 静态文件效率问题，换成nginx

## 如何部署运行？

- 首先服务器挂载cifs
- 本地build镜像 `make build`
- 导出镜像到磁盘，rsync到服务器  `make sync`
- 服务器 `docker load` 镜像
- 服务器执行镜像，可以参见 `make run-prd` 里面的命令
