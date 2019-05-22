# coding: utf-8
'''
这里是各种序列化大型numpy的方式的性能测试
目前结论：
cPickle.dumps(a, protocol=cPickle.HIGHEST_PROTOCOL) 比 cPickle.dumps(a) 快 30-50倍，和 protobuf方式不相上下
最快的是numpy的tobytes和frombuffer，但是只能用于固定shape的数据
目前选用cPickle.dumps(a, protocol=cPickle.HIGHEST_PROTOCOL)做序列化
'''
from __future__ import absolute_import, division, generators, nested_scopes, print_function, unicode_literals, with_statement

import numpy as np
import six.moves.cPickle as cPickle  # pylint: disable=no-name-in-module,import-error
import aisdk.proto as pb


def dumps_ndarray(d):
    assert isinstance(d, np.ndarray)
    return pb.NumpyNdarray(
        dtype=np.dtype(d.dtype).str, shape=d.shape,
        data=d.tobytes()).SerializeToString()


def loads_ndarray(buf):
    msg = pb.NumpyNdarray()
    msg.ParseFromString(buf)
    return np.frombuffer(
        msg.data, dtype=np.dtype(msg.dtype)).reshape(msg.shape)


def test_serialize_ndarray():
    a = np.random.rand(5, 10)
    buf = dumps_ndarray(a)
    b = loads_ndarray(buf)
    assert np.array_equiv(a, b)


benchmark_arr = np.random.rand(1080, 720, 3)


def test_serialize_ndarray_dumps(benchmark):
    a = benchmark_arr

    def dumps():
        dumps_ndarray(a)

    benchmark(dumps)


def test_serialize_ndarray_loads(benchmark):
    a = benchmark_arr
    buf = dumps_ndarray(a)

    def loads():
        loads_ndarray(buf)

    benchmark(loads)


def test_serialize_cpickle_dumps(benchmark):
    a = benchmark_arr

    def dumps():
        cPickle.dumps(a)

    benchmark(dumps)


def test_serialize_cpickle_loads(benchmark):
    a = benchmark_arr
    buf = cPickle.dumps(a)

    def loads():
        cPickle.loads(buf)

    benchmark(loads)


def test_serialize_cpickle_dumps2(benchmark):
    a = benchmark_arr

    def dumps():
        cPickle.dumps(a, protocol=cPickle.HIGHEST_PROTOCOL)

    benchmark(dumps)


def test_serialize_cpickle_loads2(benchmark):
    a = benchmark_arr
    buf = cPickle.dumps(a, protocol=cPickle.HIGHEST_PROTOCOL)

    def loads():
        cPickle.loads(buf)

    benchmark(loads)


# def test_serialize_raw_dumps(benchmark):
#     a = benchmark_arr
#     def dumps():
#         a.tobytes()
#     benchmark(dumps)

# def test_serialize_raw_loads(benchmark):
#     a = benchmark_arr
#     buf = a.tobytes()
#     def loads():
#         np.frombuffer(buf, dtype=np.float64).reshape((1080,720,3))
#     benchmark(loads)
