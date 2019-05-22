#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, generators, nested_scopes, print_function, unicode_literals, with_statement

import re
from qiniu import Auth
import requests
import os
import tarfile
import json

MODEL_DIR = "res/model"

IMAGE_CACHE_DIR = "res/testdata/image"


def read_env_file():
    env = {}
    for line in open('secret_env.sh').read().split('\n'):
        r = re.match(r'export (\w+)=(.*)', line)
        if not r:
            continue
        key = r.group(1)
        value = r.group(2)
        env[key] = value
    return env


def trim_suffix(s, suffix):
    """
    @type s: str
    @type suffix: str
    """
    assert s[-len(suffix):] == suffix
    return s[:-len(suffix)]


def download_model(model_tar_path, key=None, model_dir=None):
    """
    @type key: str
    """
    model_dir = model_dir if model_dir else MODEL_DIR
    key = model_tar_path.replace(model_dir + '/', '').replace('_', '/') \
        if not key else key
    env = read_env_file()
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if os.path.isfile(model_tar_path):
        return
    print('download model {} to {}'.format(key, model_tar_path))
    access_key = env['AVA_PUBLIC_AK']
    secret_key = env['AVA_PUBLIC_SK']
    auth = Auth(access_key, secret_key)
    domain = 'os8va9zjg.bkt.clouddn.com'
    private_url = auth.private_download_url(
        'http://{}/{}'.format(domain, key), expires=3600)
    r = requests.get(
        private_url,
        headers={'Host': domain},
        proxies={"http": 'http://nbxs-gate-io.qiniu.com:80'})
    assert r.status_code == 200
    with open(model_tar_path, 'wb') as f:
        f.write(r.content)


def extract_model_tar(model_tar_path):
    """
    @type model_tar_path: str
    """
    if not os.path.isdir(os.path.join(trim_suffix(model_tar_path, '.tar'))):
        os.mkdir(os.path.join(trim_suffix(model_tar_path, '.tar')))
    tar = tarfile.open(model_tar_path)
    tar.extractall(path=trim_suffix(model_tar_path, '.tar'))
    tar.close()


def download_model_by_app(app):
    """
    @type app: str
    """
    cfg = get_cfg_by_app(app)
    download_model(cfg['build']['model_tar'])


def get_cfg_by_app(app):
    """
    @type app: str
    """
    cfg = json.load(open('python/aisdk/app/{}/config.json'.format(app)))
    return cfg


def extract_model_by_app(app):
    """
    @type app: str
    """
    cfg = get_cfg_by_app(app)
    extract_model_tar(cfg['build']['model_tar'])


def read_tsv(key, cache_dir=None):
    """
    @type tsv_key: str
    """
    return read_test_image_with_cache(key, cache_dir=cache_dir).decode('utf8')


def parse_tsv(content):
    """
    @type content: str
    """
    return [i.split('\t') for i in content.split('\n') if len(i) > 2]


def parse_tsv_key(key):
    """
    @type tsv_key: str
    """
    return parse_tsv(read_tsv(key))


def read_test_image(key):
    """
    @type tsv_key: str
    """
    env = read_env_file()
    access_key = env['GENERAL_STORAGE_AK']
    secret_key = env['GENERAL_STORAGE_SK']
    auth = Auth(access_key, secret_key)
    domain = 'oygv408z3.bkt.clouddn.com'
    private_url = auth.private_download_url(
        'http://{}/{}'.format(domain, key), expires=3600)
    r = requests.get(
        private_url,
        headers={'Host': domain},
        proxies={"http": 'http://nbxs-gate-io.qiniu.com:80'})
    assert r.status_code == 200
    return r.content


def read_test_image_with_cache(key, cache_dir=None, cache_key=None):
    cache_dir = cache_dir if cache_dir else IMAGE_CACHE_DIR
    filename = os.path.join(cache_dir, key if not cache_key else cache_key)
    if os.path.isfile(filename):
        return open(filename, 'rb').read()
    content = read_test_image(key)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        f.write(content)
    return content
