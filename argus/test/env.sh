# 测试环境选择,可选项:product,dev
export TEST_ENV=private
#测试环境选择，可选项:stop,false
export TEST_STOP=false
# 是否下载资源
export TEST_STORE=false
# 是否打印请求
export DEBUG=true

# 测试环境选择,可选项:z0:nb;z1:bc;z2:gz;na0:lac
#z0 华东 z1 华北 z2 广州 na0 北美 
export TEST_ZONE=z0

export GOPATH=`echo $GOPATH|sed 's/^://g'`

# 用于选择app测试环境
# export TEST_APP_ENV=local
