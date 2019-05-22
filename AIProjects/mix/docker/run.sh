
docker run \
        -e 'RUN_MODE=standalone' -e 'USE_DEVICE=GPU' \
        -v '/root/song/run.cmd:/workspace/serving/run.cmd' \
        -v '/root/song/serving-eval.conf:/workspace/serving/serving-eval.conf' \
        -v '/root/song/mix.conf:/workspace/serving/mix.conf' \
        -v '/root/song/politicians:/workspace/serving/politicians' \
        -v '/root/song/ava_licence:/workspace/serving/ava_licence' \
        -v '/root/song:/root/song' \
        -v '/root/song/out/logs:/workspace/serving/out/logs' -v '/root/song/out/rets:/workspace/serving/out/rets' \
        -v '/root/song/tron_bj_run_ubuntu16.04_cuda9.0_cudnn7.1.3_engin_v0.0.2.tar:/workspace/serving/tron_bj_run_ubuntu16.04_cuda9.0_cudnn7.1.3_engin_v0.0.2.tar' \
        -v 'nvidia_driver_396.44:/usr/local/nvidia:ro' \
        -d \
        --rm \
        --network host \
        --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia7 \
        mix:20181203-final run.cmd
        # reg.qiniu.com/aiproject/mix:20180806-v2 run.cmd
        # --memory=32g --cpu-period=100000 --cpu-quota=1000000 --cpuset-cpus=1,2,3,4,5,6,7,8,9,10 \


