# TODO: remove it
from prometheus_client import CollectorRegistry, generate_latest, \
    Histogram

# pylint: disable=unexpected-keyword-arg,no-member,no-value-for-parameter,redundant-keyword-arg

REGISTRY = CollectorRegistry()


def metrics():
    return generate_latest(REGISTRY)


MONITOR_RESPONSETIME = Histogram(
    'response_time',
    'Response time of requests',
    namespace='ava',
    subsystem='serving_eval',
    labelnames=('method', 'error', 'number'),
    buckets=(
        0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        20,
        30,
        40,
        50,
        60,
    ),
    registry=REGISTRY,
)


def monitor_rt_inference(number=1, error='NIL'):
    '''Histogram ResponseTime @ inference
    '''
    return MONITOR_RESPONSETIME.labels('py.inference', error, number)


def monitor_rt_load(number=1, error='NIL'):
    '''Histogram ResponseTime @ load
    '''
    return MONITOR_RESPONSETIME.labels('py.load', error, number)


def monitor_rt_transform(number=1, error='NIL'):
    '''Histogram ResponseTime @ transform
    '''
    return MONITOR_RESPONSETIME.labels('py.transform', error, number)


def monitor_rt_forward(number=1, error='NIL'):
    '''Histogram ResponseTime @ forward
    '''
    return MONITOR_RESPONSETIME.labels('py.forward', error, number)


def monitor_rt_post(number=1, error='NIL'):
    '''Histogram ResponseTime @ post
    '''
    return MONITOR_RESPONSETIME.labels('py.post', error, number)
