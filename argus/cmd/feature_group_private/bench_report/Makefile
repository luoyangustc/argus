build:
	go build -tags 'clangcgo cublas cuda' -o bench_report

test-scripts:
	@echo "default: search 1 feature in 1M features, 50 times:"
	@echo " " LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib64/stubs ./bench_report 
	@echo "default: search 10 feature in 10M features, 500 times, cpu mode, verbose:"
	@echo " " LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib64/stubs ./bench_report -total 10 -times 500 -num 10 -verbose
	@echo "default: search 10 feature in 10M features, 500 times, gpu mode:"
	@echo " " LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib64/stubs ./bench_report -total 10 -times 500 -num 10 -verbose
