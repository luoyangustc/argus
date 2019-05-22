# tuso-cli

## 登陆(mock 不需要)

`tuso-cli login ak sk`

## 创建 hub

`tuso-cli create <Hub>`

## 添加 bucket 存量图片到 hub

1. 下载并登陆 qshell
2. `qshell listbucket <Bucket> out.txt`
3. `tuso-cli batch add <Hub> out.txt`
> 可以参考
>
> https://developer.qiniu.com/kodo/tools/1302/qshell
>
> https://github.com/qiniu/qshell/blob/master/docs/listbucket.md

