SHELL:=/bin/bash
GOREMAN:=./hack/run/bin/goreman

.PHONY:all install test lint

all: install lint test

install:
	go install ./...

test:
	TRAVIS=true go test ./...
	TRAVIS=true go test -race ./...

lint:
	@#./hack/travis-script/verify-ci.sh
	@#python2 ./hack/travis-script/verify-ci.py
	golangci-lint run

video:
	curl -o /usr/local/bin/ffmpeg  'http://s3-us-west-2.amazonaws.com/qiniu-bs/2016-05-12/ffmpeg'
	chmod +x /usr/local/bin/ffmpeg

##################################################

# Serving

# 检查serving依赖的组件是否存在
serving-check-env:
	@mkdir -p hack/run/bin && cd hack/run/bin && \
		wget -nc http://q.hi-hi.cn/goreman && \
		chmod +x goreman
	@hack/serving/check_env/check_env.sh

# 运行etcd，serving基础服务
# 如果本地有brew安装的开启自启的etcd服务，可以不使用这个
# 数据目录 hack/run/etcd-data
# 端口 2379 2380
# 运行nsqd、nsqadmin、nsqlookupd，serving基础服务
# 数据目录 hack/run/nsq-data
# 端口 nsqd：4150 4151、nsqadmin：4171、nsqlookupd：4160 4161
serving-run-base: serving-check-env
	$(GOREMAN) -p 8554 -f hack/base/Procfile start

serving-run-eval: install serving-check-env
	./hack/eval/init_etcd.sh > /dev/null
	$(GOREMAN) -p 8555 -f hack/eval/Procfile start

serving-run-tuso: install serving-check-env
	./hack/tuso/init_etcd.sh > /dev/null
	$(GOREMAN) -p 8556 -f hack/tuso/Procfile start

serving-run-fg: install serving-check-env
	./hack/face-group/init_etcd.sh > /dev/null
	$(GOREMAN) -p 8557 -f hack/face-group/Procfile start

serving-run-ig: install serving-check-env
	./hack/image-group/init_etcd.sh > /dev/null
	$(GOREMAN) -p 8558 -f hack/image-group/Procfile start

serving-run-job: install serving-check-env
	./hack/bjob/init_etcd.sh > /dev/null
	$(GOREMAN) -p 8559 -f hack/bjob/Procfile start

serving-run-dbstorage: install serving-check-env
	./hack/dbstorage/init_etcd.sh > /dev/null
	$(GOREMAN) -p 8560 -f hack/dbstorage/Procfile start

# 上面的 `$(GOREMAN) -p <port>` 端口配置要和对应 Procfile 文件里面的 `argus-hack-watch -p <port>` 一致，注意不要和其它服务组冲突