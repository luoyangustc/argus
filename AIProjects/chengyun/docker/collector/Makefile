build:
	docker build -t chenyunzhongxin-pyenv-collector:201803221540 .

all: build

run:
	docker run -it --privileged --rm -p 7757:7756 -v `pwd`:/src chenyunzhongxin-pyenv-collector:201803221540 bash

run-prd:
	docker run -d -v /mnt:/mnt --name chenyunzhongxin-pyenv-collector -p 7756:7756 chenyunzhongxin-pyenv-collector:201803221540

test:
	pytest test.py

sync:
	docker save chenyunzhongxin-pyenv-collector:201803221540 > res/chenyunzhongxin-pyenv-collector_201803221540.tar
	rsync --progress --verbose --recursive --times res/chenyunzhongxin-pyenv-collector_201803221540.tar root@10.118.61.240:/disk1/wangkechun/chenyunzhongxin-pyenv-collector_201803221540.tar
