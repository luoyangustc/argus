#coding=utf8
import multiprocessing
import os
import schema
import threading
from gevent import spawn
from aisdk.common.logger import log

default_aisdk_flavor = os.getenv("AISDK_FLAVOR", "DEV")
DEV = "DEV"
GPU_SINGLE = "GPU_SINGLE"
GPU_SHARE = "GPU_SHARE"
GPU_SHARE2 = "GPU_SHARE2"
GPU_SHARE4 = "GPU_SHARE4"

schema_flavor = schema.Schema(
    schema.Or(DEV, GPU_SHARE, GPU_SHARE2, GPU_SHARE4, GPU_SINGLE))
'''
https://jira.qiniu.io/browse/ATLAB-8694
aisdk 支持部署的时候传入一个参数 AISDK_FLAVOR，用来指定 aisdk的运行模式(部署脚本里面有默认值)，最后在启动service镜像的时候带上这个环境变量
目前值为 DEV GPU_SINGLE GPU_SHARE
DEV 为开发模式， 使用最小的实例数目，追求最低资源消耗, batch_size 也会变小
GPU_SINGLE 为单个镜像（实例）独占一张GPU卡，追求跑满GPU发挥出单卡最高性能
GPU_SHARE 为censor所有实例（大约10个app）共享一张卡，追求降低显存占用和发挥出整体最大性能
'''

from collections import namedtuple

InferenceNum = namedtuple("InferenceNum", ['n_process', 'n_thread'])


class Flavor(object):
    def __init__(self, flavor=default_aisdk_flavor, app_name=""):
        self.flavor = schema_flavor.validate(flavor)
        self.app_name = app_name
        self.forward_num = self._default_forward_num()
        self.inference_num = self._default_inference_num()

    def _default_forward_num(self):
        if self.flavor == DEV:
            return 1
        if self.flavor == GPU_SHARE:
            return 1
        if self.flavor == GPU_SHARE2:
            return 1
        if self.flavor == GPU_SHARE4:
            return 1
        if self.flavor == GPU_SINGLE:
            return 2
        raise Exception("unsupported")

    # 返回值分别为进程数和gevent线程数
    def _default_inference_num(self):
        if self.flavor == DEV:
            return InferenceNum(1, 1)
        if self.flavor == GPU_SHARE:
            return InferenceNum(12, 8)
        if self.flavor == GPU_SHARE2:
            return InferenceNum(12, 8)
        if self.flavor == GPU_SHARE4:
            return InferenceNum(12, 8)
        if self.flavor == GPU_SINGLE:
            return InferenceNum(16, 8)
        raise Exception("unsupported")

    def run_forward(self, serve_func):
        n = self.forward_num
        log.info("run_forward app_name:%s flavor:%s num:%s,", self.app_name,
                 self.flavor, n)
        if n == 1:
            t = threading.Lock()
            serve_func(t)
        elif n == 2:
            process_lock = multiprocessing.Lock()
            p1 = multiprocessing.Process(
                target=serve_func, args=(process_lock, ))
            p1.start()
            p2 = multiprocessing.Process(
                target=serve_func, args=(process_lock, ))
            p2.start()
            p1.join()
            p2.join()
        else:
            raise Exception("unsupported")

    def run_inference(self, serve_func):
        n = self.inference_num
        log.info("run_inference app_name:%s flavor:%s num:%s,", self.app_name,
                 self.flavor, n)

        def serveOneProcess():
            for _ in range(n.n_thread):
                spawn(serve_func)
            spawn(serve_func).join()

        if n.n_process == 1:
            serveOneProcess()
        else:
            ps = []
            for _ in range(n.n_process):
                p = multiprocessing.Process(target=serveOneProcess)
                p.start()
                ps.append(p)
            for p in ps:
                p.join()

    # 是否开发模式，需要使用小batch_size节约显存
    def should_small_batch(self):
        if self.flavor == DEV:
            return True
        return False
