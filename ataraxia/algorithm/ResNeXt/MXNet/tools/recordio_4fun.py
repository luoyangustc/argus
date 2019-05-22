#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import mxnet as mx
import sys


def count(file):
    r = mx.recordio.MXRecordIO(file, 'r')
    n = 0
    while True:
        item = r.read()
        if item is None:
            break
        n += 1
        if n % 1000 == 0:
            print("*", end='')
            sys.stdout.flush()
    print("\ncount of {}: {}", file, n)


def pick_step(src, dst, step):
    r = mx.recordio.MXRecordIO(src, 'r')
    w = mx.recordio.MXRecordIO(dst, 'w')
    i = 0
    n = 0
    while True:
        item = r.read()
        if item is None:
            break
        if i == 0:
            w.write(item)
            n += 1
            if n % 100 == 0:
                print("*", end='')
                sys.stdout.flush()
        i += 1
        if i >= step:
            i = 0
    print("\npick: {}".format(n))


def main(args):
    '''
    Usage:
        recordio_4fun.py count <file.rec>
        recordio_4fun.py pick <src.rec> <dst.rec> <step>
    '''
    if args['count']:
        count(args['<file.rec>'])
    elif args['pick']:
        pick_step(args['<src.rec>'], args['<dst.rec>'], int(args['<step>']))


if __name__ == "__main__":
    import docopt

    main(docopt.docopt(main.__doc__, version='0.0.1'))
