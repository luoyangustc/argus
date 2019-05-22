gen:
	protoc -I proto eval.proto --gogofast_out=plugins=grpc:golang/src/qiniu.com/ai-sdk/proto --cpp_out=cpp/proto
	python -m grpc_tools.protoc -I proto eval.proto --python_out=python/aisdk/proto
	python python/aisdk/tools/hack/fix-proto.py
	yapf -i -r python/aisdk/proto
.PHONY: gen

fmt:
	yapf -i -r .
.PHONY: fmt

install-pip-dep:
	python2 -m pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt --upgrade -t res/python-test-dep

test-all:
	./ai-sdk test pulp
	./ai-sdk test pulp_filter
	./ai-sdk test terror_detect
	./ai-sdk test terror_mixup
	./ai-sdk test terror_wangan_mixup

fmt-cpp:
	./hack/clang-format-all
