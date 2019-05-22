# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,len-as-condition,line-too-long,bad-continuation
"""bench

Usage:
  bench
  bench prepare
  bench init -c benchmark_config_path
  bench run -c benchmark_config_path [--product_version str] [--deploy_kind str] [-k name_regex] [-u int] [-d duration] [-i int] [-q] [--k6 K6_ARGS]

Options:
  -c benchmark_config_path  测试配置文件路径，必须指定
  -k name_regex  使用正则过滤测试名称，比如'censor 单op terror-all.tsv-10'代表运行'censor 单op terror'这个名字的测试，使用all.tsv数据集，只运行并发10。
  -d, --duration duration  k6 duration 测试时长 (default 30s)
  -q, --quiet    关闭k6输出
  --k6 K6_ARGS   附加的k6参数，不包括上面的三个参数

  --product_version str  产品版本，会记录在测试报告元数据里面，默认值为 unknow_product_version，如 3.0.0
  --deploy_kind str  部署方式，会记录在测试报告元数据里面，默认值为 unknow_deploy_kind，如 p4_1、p4_2、p4_4

  -h --help      Show this screen.
  --version      Show version.

使用帮助：
私有化部署性能测试工具，正常使用姿势
python2 bench.py # 查看帮助信息
python2 bench.py prepare # 下载所有测试数据集的图片到本地 benchmark_res 文件夹，同时会列出所有可用的测试配置文件
python2 bench.py init -c ./face/image_sync/benchmark/face-search/benchmark_config.yaml # 运行 人脸1:N 产品的性能测试
python2 bench.py run -c ./censor/image_sync/benchmark/censor/benchmark_config.yaml # 运行 censor 产品的性能测试
python2 bench.py run -c ./censor/image_sync/benchmark/censor/benchmark_config.yaml -d 1m # 运行 censor 产品的性能测试，时间为一分钟

如何计算测试需要的时长：目前censor3.0的qps在2-4左右，all.tsv一共12张图片，最多可能需要6s才能遍历一次数据集，推荐运行10次以上，也就是一分钟，否则会影响测试准确性

开发：
yapf -i bench.py
python3 -m pytest bench.py && python2 -m pytest bench.py
python3 -m pylint bench.py
"""
from __future__ import unicode_literals, print_function
import os
import fnmatch
import json
import subprocess
import re
import io
import time
import sys
import datetime
import string
import random
import base64
import logging as xlogging
import threading
from multiprocessing.dummy import Pool
from collections import OrderedDict
try:
    import requests
    from docopt import docopt
    import jinja2
    import yaml
    import schema
except ImportError as e:
    print(
        '请安装依赖包 sudo python2 -m pip install -i https://mirrors.aliyun.com/pypi/simple docopt requests jinja2 pyyaml schema'
    )
    raise e

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

FNULL = open(os.devnull, 'w')

VERSION = 'v0.1.8'

write_file_lock = threading.Lock()


def init_logger(name, fmt):
    l = xlogging.getLogger(name)  # type: xlogging.Logger
    l.setLevel(xlogging.DEBUG)
    formatter = xlogging.Formatter(fmt)
    handler = xlogging.StreamHandler()
    handler.setFormatter(formatter)
    l.addHandler(handler)
    return l


log = init_logger(
    'log', '%(asctime)s [%(levelname)s] %(pathname)s:%(lineno)d: %(message)s')

BENCHMARK_RES_DIR = "benchmark_res"
BENCHMARK_RES_LIST_DIR = "benchmark_res/list"
BENCHMARK_RES_IMAGE_DIR = "benchmark_res/image"
BENCHMARK_RES_RESULT_DIR = "benchmark_res/result"

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY2:
    # pylint: disable=undefined-variable
    str_type = basestring
else:
    str_type = str

REPORT_MARKDOWN_TPL = r'''
## 性能测试报告 {{meta.product_name}}-{{meta.product_version}} {{meta.start_time|strftime_timestamp}} {{meta.deploy_kind}}

> 测试工具版本：{{meta.bench_version}}， 运行参数 {{' '.join(meta.args)}}

|测试名称|测试数据集|并发|检查|QPS|响应时间(95%)|(90%)|max|min|med|arg|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
{% for result in benchmarks%}|{{result.args.name.replace('|','\|')}}|{{result.args.tsv}}|{{result.args.vus}}|{{result.result.checks}}|{{result.result.iterations}}|{{result.result.iteration_duration_p95}}|{{result.result.iteration_duration_p90}}|{{result.result.iteration_duration_max}}|{{result.result.iteration_duration_min}}|{{result.result.iteration_duration_med}}|{{result.result.iteration_duration_avg}}|
{% endfor %}
'''

benchmark_config_schema = schema.Schema({
    'product_name':
    str_type,
    'benchmarks': [{
        'name': str_type,
        'tsv': [str_type],
        'js': str_type,
        'env': {
            str_type: str_type
        },
        'vus': [int]
    }],
    schema.Optional('init'): [{
        'name':
        str_type,
        'precision':
        int,
        'dimension':
        int,
        'groups': [{
            'name': str_type,
            'port': int,
            'size': int
        }]
    }]
})

report_meta_schema = schema.Schema({
    'product_name': str_type,  # 产品名称，如 censor
    'product_version': str_type,  # 产品版本，如 3.0.0
    'start_time': float,  # 测试运行的时间戳
    'deploy_kind': str_type,  # 部署方式，如 p4_1、p4_2、p4_4
    'args': [str_type],  # 运行测试工具的参数、如 ["bench.py", "run", "-d", "10s"],
    'bench_version': str_type,  # 测试工具的版本
})

report_result_schema = schema.Schema({
    schema.Or("checks"):
    str_type,
    schema.Or("iteration_duration_p95"):
    str_type,
    schema.Or("iteration_duration_p90"):
    str_type,
    schema.Or("iteration_duration_max"):
    str_type,
    schema.Or("iteration_duration_min"):
    str_type,
    schema.Or("iteration_duration_med"):
    str_type,
    schema.Or("iteration_duration_avg"):
    str_type,
    schema.Or("iterations"):
    str_type,
})

benchmark_args_schema = schema.Schema({
    'name': str_type,
    'tsv': str_type,
    'js': str_type,
    'env': {
        str_type: str_type
    },
    'vus': int
})

# 最终报告文件的schema
report_schema = schema.Schema({
    'meta':
    report_meta_schema,
    'benchmarks': [{
        'args': benchmark_args_schema,
        'result': report_result_schema,
    }]
})


def ensure_dir(file_path):
    """
    @type filepath: str
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def normaize_path(filepath):
    """
    @type filepath: str
    """
    return filepath.lstrip('./').lstrip('/').replace('/', '_')


def url_to_filepath(url):
    """
    @type url: str
    """
    u = urlparse(url)
    path = u.path  # type: str
    return normaize_path(path)


def tsv_to_url_list(tsv_content):
    """
    @type tsv_content: str
    """
    urls = []  # type: list
    filepaths = []  # type: list
    cwd = os.getcwd()

    def url2path(url):
        return os.path.join(cwd, BENCHMARK_RES_IMAGE_DIR, url_to_filepath(url))

    for line in tsv_content.split('\n'):
        if line.startswith('#'):
            continue
        if len(line) < 2:
            continue
        fields = line.split('\t')
        if len(fields) == 1:
            assert len(fields[0]) > 0
            urls.append(fields[0])
            filepaths.append(url2path(fields[0]))
        elif len(fields) == 2:
            assert len(fields[0]) > 0
            assert len(fields[1]) > 0
            urls.append(fields[0])
            urls.append(fields[1])
            filepaths.append(url2path(fields[0]) + '\t' + url2path(fields[1]))
    return urls, filepaths


def generate_tsv_file_list(tsv_file_list):
    """
    @type tsv_file_list: list
    """
    all_urls = []
    for tsv_file_path in tsv_file_list:
        tsv_content = io.open(tsv_file_path, encoding='utf8').read()
        urls, filepaths = tsv_to_url_list(tsv_content)
        all_urls.extend(urls)
        with io.open(
                os.path.join(BENCHMARK_RES_LIST_DIR,
                             normaize_path(tsv_file_path)),
                'w',
                encoding='utf8') as f:
            f.write('\n'.join(filepaths))
    return all_urls


def download_one_image(url):
    """
    @type url: str
    """
    imgfilepath = os.path.join(BENCHMARK_RES_IMAGE_DIR, url_to_filepath(url))
    if os.path.isfile(imgfilepath):
        return
    log.info('downloading %s...', url)
    if url.startswith('http://pj5mpx90q.bkt.clouddn.com'):
        proxies = {"http": "http://nbxs-gate-io.qiniu.com:80"}
        response = requests.get(url, proxies)
    else:
        response = requests.get(url)
    assert response.status_code == 200, 'code: {} url: {}'.format(response.status_code, url)
    img = response.content
    with write_file_lock:
        with io.open(imgfilepath + '_tmp', 'wb') as f:
            f.write(img)
        os.rename(imgfilepath + '_tmp', imgfilepath)


def download_images(urls):
    """
    @type urls: list
    """
    pool = Pool(10)
    pool.map(download_one_image, urls)


def scan_tsv():
    matches = []
    for root, _, filenames in os.walk('.'):
        if root.count(BENCHMARK_RES_DIR) > 0:
            continue
        for filename in fnmatch.filter(filenames, '*.tsv'):
            matches.append(os.path.join(root, filename))
    return matches


def scan_config_yaml():
    matches = []
    for root, _, filenames in os.walk('.'):
        if root.count(BENCHMARK_RES_DIR) > 0:
            continue
        for filename in filenames:
            if filename == 'benchmark_config.yaml':
                matches.append(os.path.join(root, filename))
    return matches


def render_report(report_json_path):
    """
    @type report_json_path: str
    """
    r = json.loads(io.open(report_json_path, encoding='utf8').read())
    report_schema.validate(r)
    env = jinja2.Environment()
    env.filters[
        'strftime_timestamp'] = lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y%m%d%H%M%S')
    s = env.from_string(REPORT_MARKDOWN_TPL).render(r)
    return s


def parse_k6_data(k6_output_text):
    result = OrderedDict()
    txt_lines = k6_output_text.split('\n')
    for txt_line in txt_lines:
        if 'checks..' in txt_line:
            result['checks'] = re.search(r'(\d+(\.\d+)?\%)', txt_line).group(1)
        elif 'iterations..' in txt_line:
            result['iterations'] = re.search(r'(\S+/s)$', txt_line).group(1)
        elif 'iteration_duration..' in txt_line:
            result['iteration_duration_p95'] = re.search(
                r'p\(95\)=(\S+)', txt_line).group(1)
            result['iteration_duration_p90'] = re.search(
                r'p\(90\)=(\S+)', txt_line).group(1)
            result['iteration_duration_max'] = re.search(
                r'max=(\S+)', txt_line).group(1)
            result['iteration_duration_min'] = re.search(
                r'min=(\S+)', txt_line).group(1)
            result['iteration_duration_med'] = re.search(
                r'med=(\S+)', txt_line).group(1)
            result['iteration_duration_avg'] = re.search(
                r'avg=(\S+)', txt_line).group(1)
    report_result_schema.validate(result)
    return result


def prepare():
    ensure_dir(BENCHMARK_RES_DIR + '/')
    ensure_dir(BENCHMARK_RES_LIST_DIR + '/')
    ensure_dir(BENCHMARK_RES_IMAGE_DIR + '/')
    ensure_dir(BENCHMARK_RES_RESULT_DIR + '/')
    log.info("scan config json")
    config_yaml_list = scan_config_yaml()
    log.info("config_yaml_list %s", json.dumps(config_yaml_list))
    log.info("scan tsv")
    tsv_file_list = scan_tsv()
    log.debug("tsv_file_list %s", tsv_file_list)
    log.info("generate file list for %s tsv", len(tsv_file_list))
    urls = generate_tsv_file_list(tsv_file_list)
    log.info("download %s image", len(urls))
    download_images(urls)
    log.info("download image success")


def face_group_add_feature(url, size, length):
    # 插入fake人脸特征
    features = []
    id = ""
    for _ in range(15):
        id += random.choice(string.ascii_letters + string.digits)
    arrb = bytearray([0] * length)
    for j in range(length):
        arrb[j] = random.randint(1, 9)
    feature_base64 = base64.encodestring(arrb).decode('ascii')
    for k in range(size):
        features.append({
            "id": id + str(k),
            "value": feature_base64,
            "tag": id + str(k),
        })
    requests.post(
        url,
        data=json.dumps({"features": features}),
        headers={"Content-Type": "application/json"})


def face_group(group, dimension, precision):
    name = group['name']
    port = group['port']
    size = group['size']
    # 创建group
    url = "http://localhost:" + str(port) + "/v1/face/groups/" + name
    data = {'config': {"capacity": size}}
    requests.post(
        url,
        data=json.dumps(data),
        headers={"Content-type": "application/json"})

    # 添加fake人脸特征
    url = "http://localhost:" + str(
        port) + "/v1/face/groups/" + name + "/feature/add"
    batch = 1000
    for i in range(size // batch):
        face_group_add_feature(url, batch, precision * dimension)
        log.info("add %d-%d features into group %s", i * batch,
                 (i + 1) * batch - 1, name)
    remain = size % batch
    if remain > 0:
        face_group_add_feature(url, remain, precision * dimension)
        log.info("add %d-%d features into group %s", size - remain, size, name)


def init_face_search(conf):
    # 初始化人脸搜索底库
    groups = conf["groups"]
    for group in groups:
        face_group(group, conf["precision"], conf["dimension"])


def init(config_file):
    # yaml验证
    config = benchmark_config_schema.validate(
        yaml.load(io.open(config_file, encoding='utf8')))
    for item in config['init']:
        if item['name'] == 'face-search':
            init_face_search(item)


class RunCmdException(Exception):
    pass


def run(cmd, quiet=False):
    if quiet:
        p = subprocess.Popen(
            cmd,
            stdout=FNULL,
            stderr=FNULL,
            shell=True,
            executable='/bin/bash')
    else:
        print('- ', cmd)
        p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    p.wait()
    if p.returncode != 0:
        raise RunCmdException()


def get_k6_env_arg(env):
    """
    @type env: dict
    """
    r = []
    for k, v in env.items():
        r.append('-e "{}={}"'.format(k, v))
    return ' '.join(r)


def run_benchmark(config_json, benchmark_args, k6_duration, k6_args, quiet):
    benchmark_args_schema.validate(benchmark_args)
    log.info("run k6 for %s", json.dumps(benchmark_args, ensure_ascii=False))
    tsv_abs_path = os.path.join(
        os.getcwd(), BENCHMARK_RES_LIST_DIR,
        normaize_path(
            os.path.join(os.path.dirname(config_json), benchmark_args['tsv'])))
    out_json_path = os.path.join(os.getcwd(), BENCHMARK_RES_RESULT_DIR,
                                 'tmp.json')
    out_txt_path = os.path.join(os.getcwd(), BENCHMARK_RES_RESULT_DIR,
                                'tmp.txt')
    k6js_path = os.path.join(
        os.path.dirname(config_json), benchmark_args['js'])
    cmd = '''k6 run \
-e BENCHMARK_DATA_LIST_FILE={tsv_abs_path} \
{k6_env_arg} \
--vus={k6_vus} --duration={k6_duration} \
--out json={out_json_path} {k6_args} \
{k6js_path} | tee {out_txt_path}'''
    run(cmd.format(
        tsv_abs_path=tsv_abs_path,
        out_json_path=out_json_path,
        out_txt_path=out_txt_path,
        k6js_path=k6js_path,
        k6_args=k6_args,
        k6_vus=benchmark_args['vus'],
        k6_duration=k6_duration,
        k6_env_arg=get_k6_env_arg(benchmark_args['env']),
    ),
        quiet=quiet)
    result = parse_k6_data(io.open(out_txt_path, encoding='utf8').read())
    log.info('test result %s %s', json.dumps(
        benchmark_args, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False))
    return result


def run_all(  # pylint: disable=too-many-arguments,too-many-locals
        config_yaml, k6_duration, k6_args, quiet, name_regex, product_version,
        deploy_kind):
    if k6_duration is None:
        k6_duration = '60s'
    if k6_args is None:
        k6_args = ''
    if product_version is None:
        product_version = 'unknow_product_version'
    if deploy_kind is None:
        deploy_kind = 'unknow_deploy_kind'
    benchmark_results = []
    start_time = time.time()

    # yaml验证
    config = benchmark_config_schema.validate(
        yaml.load(io.open(config_yaml, encoding='utf8')))

    # 开始运行
    try:
        for benchmark_set in config['benchmarks']:
            name = benchmark_set['name']
            tsv_list = benchmark_set['tsv']
            vus_list = benchmark_set['vus']
            for tsv in tsv_list:
                for vus in vus_list:
                    if name_regex and (not re.search(
                            name_regex.decode('utf8'), '{}-{}-{}'.format(
                                name, tsv, vus))):
                        continue
                    benchmark_args = {
                        'name': name,
                        'tsv': tsv,
                        'js': benchmark_set['js'],
                        'env': benchmark_set['env'],
                        'vus': vus,
                    }
                    result = run_benchmark(config_yaml, benchmark_args,
                                           k6_duration, k6_args, quiet)
                    benchmark_args_schema.validate(benchmark_args)
                    benchmark_results.append({
                        'args': benchmark_args,
                        'result': result
                    })
    except KeyboardInterrupt:
        pass

    meta = report_meta_schema.validate({
        'product_name': config['product_name'],
        'product_version': product_version,
        'start_time': start_time,
        'deploy_kind': deploy_kind,
        'args': sys.argv,
        'bench_version': VERSION,
    })
    report = {'meta': meta, 'benchmarks': benchmark_results}
    report_schema.validate(report)
    report_path = os.path.join(BENCHMARK_RES_RESULT_DIR, 'report.json')
    log.info('write config to %s', report_path)
    with open(report_path, 'w') as f:
        json.dump(report, f)

    print('\n\n\n----------------')
    print(render_report(report_path))


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    current_dir = os.getcwd()
    log.info("current_dir %s", current_dir)
    arg = docopt(__doc__, version=VERSION)
    if os.getenv('DEBUG') is not None:
        log.setLevel(xlogging.DEBUG)
        log.info('use debug log')
    else:
        log.setLevel(xlogging.INFO)
    log.debug('args %s', arg)
    if arg['prepare']:
        prepare()
    if arg['init']:
        config_yaml = arg['-c']
        init(config_yaml)
    elif arg['run']:
        prepare()
        run_all(
            config_yaml=arg['-c'],
            k6_duration=arg['--duration'],
            k6_args=arg['--k6'],
            quiet=arg['--quiet'],
            name_regex=arg['-k'],
            product_version=arg['--product_version'],
            deploy_kind=arg['--deploy_kind'])
    else:
        print(__doc__)


if __name__ == '__main__':
    main()


def test_normaize_path():
    assert normaize_path(
        'censor/image_sync/benchmark/censor/politician-normal.tsv'
    ) == 'censor_image_sync_benchmark_censor_politician-normal.tsv'
    assert normaize_path('/a/b') == 'a_b'
    assert normaize_path('./a/b') == 'a_b'


def test_tsv_to_file_list():
    file_list = scan_tsv()
    file_list.sort()
    tsv_content = io.open(file_list[0], encoding='utf8').read()
    urls, filepaths = tsv_to_url_list(tsv_content)
    assert urls[
        0] == 'http://pj5mpx90q.bkt.clouddn.com/politician-normal/politician-normal-1.jpg'
    assert normaize_path(url_to_filepath(
        urls[0])) == 'politician-normal_politician-normal-1.jpg'
    path = filepaths[0]  # type: str
    assert path[0] == '/'


def test_parse_k6_data():
    result = parse_k6_data('''
  execution: local
     output: json=/home/qnai/wangkechun/src/qiniu.com/argus/service/service/image/benchmark_res/result/tmp.json
     script: censor/image_sync/benchmark/censor/k6.js

    duration: 30s, iterations: -
         vus: 10,  max: 10

    init [----------------------------------------------------------] starting
    ✓ is status 200

    checks.....................: 100.00% ✓ 78   ✗ 0
    data_received..............: 42 kB   1.4 kB/s
    data_sent..................: 18 MB   614 kB/s
    http_req_blocked...........: avg=84.96µs  min=2.85µs   med=4.97µs   max=3.26ms   p(90)=245.37µs p(95)=341.2µs
    http_req_connecting........: avg=51.08µs  min=0s       med=0s       max=2.71ms   p(90)=109.65µs p(95)=146.83µs
    http_req_duration..........: avg=3.56s    min=507.73ms med=3.83s    max=7.27s    p(90)=4.37s    p(95)=4.41s
    http_req_receiving.........: avg=109.19µs min=67.39µs  med=96.44µs  max=454.87µs p(90)=138.74µs p(95)=163.29µs
    http_req_sending...........: avg=425.16µs min=74.13µs  med=295.64µs max=2.58ms   p(90)=791.18µs p(95)=1ms
    http_req_tls_handshaking...: avg=0s       min=0s       med=0s       max=0s       p(90)=0s       p(95)=0s
    http_req_waiting...........: avg=3.56s    min=506.83ms med=3.83s    max=7.27s    p(90)=4.37s    p(95)=4.41s
    http_reqs..................: 78      2.599994/s
    iteration_duration.........: avg=3.57s    min=526.96ms med=3.83s    max=7.28s    p(90)=4.39s    p(95)=4.42s
    iterations.................: 78      2.599994/s
    vus........................: 10      min=10 max=10
    vus_max....................: 10      min=10 max=10

''')
    assert result == {
        'iteration_duration_p95': '4.42s',
        'iteration_duration_p90': '4.39s',
        'iteration_duration_max': '7.28s',
        'iteration_duration_min': '526.96ms',
        'iteration_duration_med': '3.83s',
        'iteration_duration_avg': '3.57s',
        'iterations': '2.599994/s',
        'checks': '100.00%'
    }


def test_get_k6_env_arg():
    env = {"scenes": "pulp|terror|politician"}
    assert get_k6_env_arg(env) == '-e "scenes=pulp|terror|politician"'


def test_render_report():
    md = render_report('./testdata/report.json')
    md2 = io.open('./testdata/report.md', encoding='utf8').read()
    assert md == md2


def test_face_group_add_feature():
    try:
        face_group_add_feature("http://localhost:9999", 1000, 2048)
    except requests.ConnectionError:
        pass
