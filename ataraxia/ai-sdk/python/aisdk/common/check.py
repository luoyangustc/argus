from .error import ErrorConfig, ErrorFileNotExist
import os


def file_check(configs, key, must=True):
    '''
        get file with check
    '''
    target = ""
    if key not in configs:
        if must:
            raise ErrorConfig(key)
        return None
    target = configs[key]
    if not os.path.isfile(target):
        raise ErrorFileNotExist(target)
    return target


def value_check(configs, key, must=True, default=None):
    '''
        get value with check
    '''
    if key not in configs or configs[key] is None:
        if not must:
            return default
        raise ErrorConfig(key)
    return configs[key]
