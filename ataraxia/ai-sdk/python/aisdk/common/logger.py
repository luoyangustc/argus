import logging as xlogging


# pylint: disable=invalid-name
def init_logger(name, fmt):
    l = xlogging.getLogger(name)  # type: xlogging.Logger
    l.setLevel(xlogging.DEBUG)
    formatter = xlogging.Formatter(fmt)
    handler = xlogging.StreamHandler()
    handler.setFormatter(formatter)
    l.addHandler(handler)
    return l


xl = init_logger(
    'xlog',
    '%(asctime)s [%(levelname)s] [%(reqid)s] %(pathname)s:%(lineno)d: %(message)s'
)

log = init_logger(
    'log', '%(asctime)s [%(levelname)s] %(pathname)s:%(lineno)d: %(message)s')
