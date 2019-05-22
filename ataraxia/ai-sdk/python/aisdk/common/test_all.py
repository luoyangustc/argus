# pylint: disable=unused-import
import os
import os.path
import pytest

from . import check
from . import error
from . import image
from . import logger
from . import monitor
from . import other
from . import config
from . import flavor


def test_file_check():
    old_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    assert check.file_check({
        'test_file': 'test_all.py'
    }, 'test_file') == 'test_all.py'
    os.chdir(old_dir)


def test_value_check():
    assert check.value_check({
        'test_file': 'test_all.py'
    }, 'test_file') == 'test_all.py'


def test_logger():
    logger.log.info('hello')
    logger.xl.info('hello with reqid', extra={'reqid': 'xxxx'})


def test_load_config_success():
    config.load_config("pulp")


def test_load_config_change_batch_size():
    cfg = config.load_config("pulp")
    assert cfg['batch_size'] == 8
    f = flavor.Flavor('DEV')
    cfg2 = config.load_config("pulp", flavor=f)
    assert cfg2['batch_size'] == 2


def test_flavor():
    with pytest.raises(Exception):
        flavor.Flavor("xx")
    dev = flavor.Flavor("DEV")
    assert dev.forward_num == 1
    assert dev.inference_num == flavor.InferenceNum(1, 1)

    dev = flavor.Flavor("GPU_SINGLE")
    assert dev.forward_num == 2
    assert dev.inference_num == flavor.InferenceNum(16, 8)
