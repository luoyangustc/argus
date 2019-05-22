## Callback 测试服务

####我们在cd1上部署了回调测试服务，以方便一些功能测试，比如验证上传回调功能等

服务可执行文件地址：/home/qboxserver/qiniu_callback_test_server
服务监听端口： 8090

####添加或跟新服务流程：

1. Coding，完成你的需求
2. 编译项目，注：如果在Mac上，应使用下面命令，以编译符合Linux要求的二进制文件

   ```
   GOOS=linux go build
   ```
3. 接下来我们就要把编译出的文件拷贝到cd1上，然后启动它，但是因为旧的服务已经在运行，我们首先需要把它停掉，然后才好部署新的服务:

   ```
   qboxserver@cd1:~$ ps aux | grep qiniu_callback_test_server
   1001      9728  0.0  0.0   8744   956 pts/0    S+   10:40   0:00 grep --color=auto qiniu_callback_test_server
   1001     48986  0.0  0.0   8512  2872 ?        Sl   10:04   0:00 ./qiniu_callback_test_serve -addr=0.0.0.0:8090
   qboxserver@cd1:~$ kill 48986
   ```

3. 把文件copy到cd1，假设上一步编译出的可执行文件叫 qiniu_callback_test_server， 则：

   ```
   scp qiniu_callback_test_server qboxserver@cd1:~
   ```
   
4. 启动服务：

   ```
   qboxserver@cd1:~$ nohup ./qiniu_callback_test_server -addr=0.0.0.0:8090 &
   [1] 10767 nohup: ignoring input and appending output to `nohup.out'
   ```

5. 测试下，服务是否启动成功。