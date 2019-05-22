# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,len-as-condition,line-too-long,bad-continuation
"""bench

Usage:
  bench
  bench prepare
  bench init -c benchmark_config_path
  bench run -c benchmark_config_path [--product_version str] [--deploy_kind str]

Options:
  -c benchmark_config_path 测试配置文件路径，必须指定

  --product_version str  产品版本，会记录在测试报告元数据里面，默认值为 unknow_product_version，如 3.0.0
  --deploy_kind str  部署方式，会记录在测试报告元数据里面，默认值为 unknow_deploy_kind，如 p4_1、p4_2、p4_4

  -h --help      Show this screen.
  --version      Show version.

使用帮助：
私有化部署性能测试工具，正常使用姿势
python2 bench.py # 查看帮助信息
python2 bench.py prepare # 下载所有测试数据集的视频到本地 benchmark_res 文件夹，同时会列出所有可用的测试配置文件
python2 bench.py init -c ./face_search/video/benchmark/live/benchmark_config.yaml # 运行 人脸1:N 产品的性能测试
python2 bench.py run -c ./censor/image_sync/benchmark/censor/benchmark_config.yaml # 运行 censor 产品的性能测试
"""
from __future__ import unicode_literals, print_function
import fnmatch
import base64
import cgi
import datetime
import io
import sys
import os
import json
import random
import string
import time
import logging as xlogging
import threading
from multiprocessing.dummy import Pool
try:
    import cv2
    import requests
    import jinja2
    from docopt import docopt
    import yaml
    import schema
except ImportError as e:
    print(
        '请安装依赖包 sudo python2 -m pip install -i https://mirrors.aliyun.com/pypi/simple docopt requests pyyaml schema opencv-python'
    )
    raise e

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

VERSION = 'v0.0.1'

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

BENCHMARK_HOST = "127.0.0.1"
BENCHMARK_RES_DIR = "benchmark_res"
BENCHMARK_RES_LIST_DIR = "benchmark_res/list"
BENCHMARK_RES_NGINX_ADDRESS = "http://{}:8900/test".format(BENCHMARK_HOST)
BENCHMARK_RES_VIDEO_NGINX_DIR = os.path.join(os.environ['HOME'],
                                             "ava_workspace/data/nginx_/test")
BENCHMARK_RES_VIDEO_DIR = "benchmark_res/video"
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

|测试名称|测试数据集|采样帧距|采样时间间隔|最大并发流数|
|-----|-----|-----|-----|-----|
{% for result in benchmarks%}|{{result.args.name.replace('|','\|')}}|{{result.args.tsv}}|{{result.args.interval}}|{{result.args.interval_time}}|{{result.result.streams}}|
{% endfor %}
'''

# benchmark_config.yaml schema
benchmark_config_schema = schema.Schema({
    'product_name':
    str_type,
    'benchmarks': [{
        'name': str_type,
        'tsv': [str_type],
        'env':{
            str_type: str_type
        },
        'interval': [int]
    }],
    'init': [{
        'name': str_type,
        'precision': int,
        'dimension': int,
        'groups': [{
            'name': str_type,
            'port': int,
            'size': int
        }]
    }]
})

benchmark_args_schema = schema.Schema({
    'name': str_type,
    'tsv': str_type,
    'env': {
        str_type: str_type
    },
    'interval': int,
    'interval_time': float
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
    'resolution': str_type,
    'interval': int,
    'streams': int,
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

    def url2localurl(url):
        return os.path.join(BENCHMARK_RES_NGINX_ADDRESS,
                            BENCHMARK_RES_VIDEO_DIR, url_to_filepath(url))

    for line in tsv_content.split('\n'):
        if line.startswith('#'):
            continue
        if len(line) < 2:
            continue
        fields = line.split('\t')
        if len(fields) == 1:
            assert len(fields[0]) > 0
            urls.append(fields[0])
            filepaths.append(url2localurl(fields[0]))
        elif len(fields) == 2:
            assert len(fields[0]) > 0
            assert len(fields[1]) > 0
            urls.append(fields[0])
            urls.append(fields[1])
            filepaths.append(
                url2localurl(fields[0]) + '\t' + url2localurl(fields[1]))
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


def download_one_video(url):
    """
    @type url: str
    """
    videooath = os.path.join(BENCHMARK_RES_VIDEO_NGINX_DIR,
                             BENCHMARK_RES_VIDEO_DIR, url_to_filepath(url))
    if os.path.isfile(videooath):
        return
    log.info('downloading %s...', url)
    response = requests.get(url)
    video = response.content
    with write_file_lock:
        with io.open(videooath + '_tmp', 'wb') as f:
            f.write(video)
        os.rename(videooath + '_tmp', videooath)


def download_videos(urls):
    """
    @type urls: list
    """
    video_dir = os.path.join(BENCHMARK_RES_VIDEO_NGINX_DIR,
                             BENCHMARK_RES_VIDEO_DIR)
    if not os.path.isdir(video_dir):
        os.makedirs(video_dir)
    pool = Pool(10)
    pool.map(download_one_video, urls)


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


def start_live(url, vid, interval, group, port):
    data = {
        "data": {
            "uri": url
        },
        "params": {
            "vframe": {
                "mode":0,
                "interval": interval
            },
            "live": {
                "timeout": 30,
                "downstream": "rtsp://{}:58080/live/{}".format(BENCHMARK_HOST,vid)
            }
        },
        "ops":[
            {
                "op":"face_group_search_private",
                "params": {
                    "other":{
                        "groups": [group],
                        "threshold": 0.35,
                        "limit": 1
                    },
                "ignore_empty_labels": False
                }
            }
        ]
    }
    url = "http://{}:{}/v1/video/{}/async".format(BENCHMARK_HOST, port, vid)
    resp = requests.post(
        url,
        data=json.dumps(data),
        headers={"Content-type": "application/json"})
    if resp.status_code == 200:
        return resp.json()['job']
    else:
        log.error("start live error: %s", resp.status_code)
    return ''


def get_live_doing(port):
    url = "http://{}:{}/v1/jobs".format(BENCHMARK_HOST, port)
    jobs = []
    resp = requests.get(url, params={'status': 'Cancelling'})
    if resp.status_code == 200:
        jobs = resp.json()['jobs']
    else:
        log.error("gett_live_doing: %s", resp.status_code)
    return jobs


def stop_live(jobs, port):
    for job in jobs:
        url = "http://{}:{}/v1/jobs/{}/kill".format(BENCHMARK_HOST, port, job)
        resp = requests.post(url)
        if resp.status_code != 200:
            log.error("stop %s failed, error %d", job, resp.status_code)
            return
    while True:
        if len(get_live_doing(port)) == 0:
            break
        time.sleep(1)


def get_response_time_and_count(port):
    resp = requests.get("http://{}:{}/metrics".format(BENCHMARK_HOST, port))
    if resp.status_code != 200:
        log.error("get metrics failed, error %d", resp.status_code)
        return
    count = '0'
    response_time = '0'
    for line in resp.text.split("\n"):
        if "qiniu_ai_video_eval_response_time_count" in line:
            count = line.split(" ")[1]
        elif "qiniu_ai_video_eval_response_time_sum" in line:
            response_time = line.split(" ")[1]
    return count, response_time


def cap(port):
    ori_count, ori_resp = get_response_time_and_count(port)
    time.sleep(30)
    count, resp = get_response_time_and_count(port)
    if count == '0' or resp == '0':
        return 0
    return (float(resp)-float(ori_resp))/(float(count) - float(ori_count))


def run_benchmark(config_json, benchmark_args, last_streams):
    tsv_abs_path = os.path.join(os.getcwd(), BENCHMARK_RES_LIST_DIR,
                                benchmark_args['tsv'])
    tsv_content = io.open(tsv_abs_path, encoding='utf8').read()
    videos = tsv_content.split('\n')
    jobs = []
    vid = "video_benchmark_{}_{}".format(benchmark_args['interval'], 0)
    job = start_live(random.choice(videos), vid, benchmark_args['interval'],
                     benchmark_args['env']['group'], benchmark_args['env']['port'])
    if job != '':
        jobs.append(job)
        log.info("video %s, interval %d, start job %s, total streams %d",
                 benchmark_args['tsv'],
                 benchmark_args['interval'],
                 job, len(jobs))
    else:
        return 0, 0

    if last_streams > 1:
        for i in range(last_streams-1):
            time.sleep(5)
            vid = "video_benchmark_{}_{}".format(benchmark_args['interval'], len(jobs))
            job = start_live(random.choice(videos), vid, benchmark_args['interval'],
                             benchmark_args['env']['group'], benchmark_args['env']['port'])
            if job != '':
                jobs.append(job)
                log.info("video %s, interval %d, start job %s, total streams %d, last delay: %fs",
                         benchmark_args['tsv'],
                         benchmark_args['interval'],
                         job, len(jobs), 0)
            else:
                stop_live(jobs, benchmark_args['env']['port'])
                return 0, 0

    for i in range(100):
        while True:
            delay = cap(benchmark_args['env']['port'])
            if delay == 0:
                time.sleep(1)
                continue
            threshold = float(benchmark_args['interval'])/25.0
            if threshold > 0.5 and threshold <= 2:
                threshold = float(benchmark_args['interval'])/(25.0*2)
            elif threshold > 2:
                threshold = 1
            if delay > threshold:
                stop_live(jobs, benchmark_args['env']['port'])
                return len(jobs), delay
            break

        for i in range(benchmark_args['interval']//25+1):
            vid = "video_benchmark_{}_{}".format(benchmark_args['interval'], len(jobs))
            job = start_live(random.choice(videos), vid, benchmark_args['interval'],
                             benchmark_args['env']['group'], benchmark_args['env']['port'])
            if job != '':
                jobs.append(job)
                log.info("video %s, interval %d, start job %s, total streams %d, last delay: %fs",
                         benchmark_args['tsv'],
                         benchmark_args['interval'],
                         job, len(jobs), delay)
            else:
                stop_live(jobs, benchmark_args['env']['port'])
                return 0, 0


def run_all(config_yaml, product_version, deploy_kind):

    # yaml验证
    config = benchmark_config_schema.validate(
        yaml.load(io.open(config_yaml, encoding='utf8')))

    # 开始运行
    start_time = time.time()
    benchmark_results = []
    try:
        streams = 0
        for benchmark_set in config['benchmarks']:
            name = benchmark_set['name']
            tsv_list = benchmark_set['tsv']
            interval_list = benchmark_set['interval']
            for tsv in tsv_list:
                for interval in interval_list:
                    benchmark_args = {
                        'name': name,
                        'tsv': tsv,
                        'env': benchmark_set['env'],
                        'interval': interval,
                        "interval_time": interval/25.0
                    }
                    benchmark_args_schema.validate(benchmark_args)
                    streams, delay = run_benchmark(config_yaml, benchmark_args, streams)
                    if streams == 0:
                        log.error("fail to run live benchmark")
                    elif streams > 1:
                        streams -= (interval//25+1)
                    result = {
                        "resolution": tsv,
                        "interval": interval,
                        "streams": streams}
                    report_result_schema.validate(result)
                    benchmark_args_schema.validate(benchmark_args)
                    benchmark_results.append({
                        'args': benchmark_args,
                        'result': result
                    })
                streams = 0
                            
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
    url = "http://{}:".format(BENCHMARK_HOST) + str(port) + "/v1/face/groups/" + name
    data = {'config': {"capacity": size}}
    requests.post(
        url,
        data=json.dumps(data),
        headers={"Content-type": "application/json"})

    # 添加fake人脸特征
    url =  "http://{}:".format(BENCHMARK_HOST) + str(
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


def prepare():
    ensure_dir(BENCHMARK_RES_DIR + '/')
    ensure_dir(BENCHMARK_RES_LIST_DIR + '/')
    ensure_dir(BENCHMARK_RES_VIDEO_DIR + '/')
    ensure_dir(BENCHMARK_RES_RESULT_DIR + '/')
    log.info("scan config json")
    config_yaml_list = scan_config_yaml()
    log.info("config_yaml_list %s", json.dumps(config_yaml_list))
    log.info("scan tsv")
    tsv_file_list = scan_tsv()
    log.debug("tsv_file_list %s", tsv_file_list)
    log.info("generate file list for %s tsv", len(tsv_file_list))
    urls = generate_tsv_file_list(tsv_file_list)
    log.info("download %s video", len(urls))
    download_videos(urls)
    log.info("download video success")


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
        run_all(
            config_yaml=arg['-c'],
            product_version=arg['--product_version'],
            deploy_kind=arg['--deploy_kind'])
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
