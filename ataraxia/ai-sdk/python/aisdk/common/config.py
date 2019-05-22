import schema
import json
import os
import os.path
import six
from .flavor import Flavor
from .logger import log

str_type = six.string_types[0]

cfg_schema = schema.Schema({
    'app_name': str_type,
    'build': {
        'model_tar': str_type
    },
    'model_files': {
        str_type: str_type
    },
    'batch_size': int,
    schema.Optional('image_width'): int,
    'use_device': schema.Or('GPU', 'CPU'),
    schema.Optional('custom_params'): {
        str_type: object
    }
})


def load_config(app_name, flavor=None):
    """
    @type app_name: str
    @type flavor: Flavor
    """
    cfg = cfg_schema.validate(
        json.loads(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), '../app/',
                    app_name, 'config.json')).read()))
    app_name = cfg['app_name']
    assert cfg['app_name'] == app_name
    if flavor != None:
        assert isinstance(flavor, Flavor)
        if flavor.should_small_batch():
            if cfg['batch_size'] > 2:
                old_batch_size = cfg['batch_size']
                cfg['batch_size'] = 2
                log.info("change batch_size for dev, %s -> %s", old_batch_size,
                         cfg['batch_size'])
    return cfg
