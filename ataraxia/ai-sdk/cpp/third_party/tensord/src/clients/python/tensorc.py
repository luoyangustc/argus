"""Tensor Client.

Usage:
  tensorc.py predict [--grpc=<string>] \
                     --model=<string> [--model_version=<int>] \
                     --in=<name=string>... \
                     [--out=<name=@string>...]
  tensorc.py predict_batch [--grpc=<string>] \
                           --model=<string> [--model_version=<int>] \
                           [--batch_size=<int>] \
                           [--in=<name=string>...] [--out=<name=string>] \
                           <input_dir> [<output_dir>]
  tensorc.py (-h | --help)
  tensorc.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --grpc=<grpc_address>           gRPC address [default: localhost:50001]
  --model=<model>                 Model name
  --model_version=<model_version> Model version [default: 0]

"""
import base64
from docopt import docopt
import grpc
import time
import os

import tensord_pb2_grpc
import tensord_pb2


def readInput(input):
    i = input.find('=')
    if input[i+1] == '@':
        with open(input[i+2:], 'r') as file:
            return input[:i], file.read()
    return input[:i], base64.b64decode(input[i+1:])


def parseAlias(output):
    i = output.find('=')
    return (output, output) if i == -1 else (output[:i], output[i+1:])


def predict(args):

    model = args["--model"]
    version = args['--model_version']
    if version is None:
        version = 0

    req = tensord_pb2.Requests()
    req.model = model
    req.version = version

    request = req.request.add()
    for input in args['--in']:
        data = request.data.add()
        data.datatype = tensord_pb2.FLOAT32
        data.name, data.body = readInput(input)

    grpc_address = args.get("--grpc", "localhost:50001")
    channel = grpc.insecure_channel(grpc_address)
    stub = tensord_pb2_grpc.TensordStub(channel)

    t1 = time.time()
    resp = stub.Predict(req)
    t2 = time.time()

    response = resp.response[0]
    for data in response.data:
        found = False
        for output in args['--out']:
            name, alias = parseAlias(output)
            if name == data.name:
                with open(alias, 'w+') as file:
                    file.write(data.body)
                found = True
                break
        if not found:
            print data.name, data.body

    print 'predict: -----'
    print 'model: {}, version: {}'.format(model, version)
    print 'request size: ', req.ByteSize()
    print 'response size: ', resp.ByteSize()
    print 'response time: {:.2f}ms'.format((t2 - t1) * 1000)

    channel.close()


def predict_batch(args):

    model = args["--model"]
    version = args['--model_version']
    if version is None:
        version = 0
    batch_size = args['--batch_size']
    if batch_size is None:
        batch_size = 1
    input_dir = args['<input_dir>']
    output_dir = args['<output_dir>']
    if output_dir is None:
        output_dir = input_dir + "_out"

    input_alias = {}
    for input in args['--in']:
        name, alias = parseAlias(input)
        input_alias[name] = alias
    output_alias = {}
    for output in args['--out']:
        name, alias = parseAlias(output)
        output_alias[name] = alias

    grpc_address = args.get("--grpc", "localhost:50001")
    channel = grpc.insecure_channel(grpc_address)
    stub = tensord_pb2_grpc.TensordStub(channel)

    metrics = [0.0, 0]

    def predict(names, req, metrics=[0.0, 0]):
        t1 = time.time()
        resp = stub.Predict(req)
        t2 = time.time()

        print 'predict: batch_size={} request_size={} response_size={} response_time={:.2f}ms'.format(
            len(names), req.ByteSize(), resp.ByteSize(), (t2 - t1) * 1000)

        metrics[0] += t2 - t1
        metrics[1] += len(names)

        for i, response in enumerate(resp.response):
            for data in response.data:
                if not os.path.isdir(os.path.join(output_dir, names[i])):
                    os.makedirs(os.path.join(output_dir, names[i]))
                filename = os.path.join(output_dir, names[i],
                                        output_alias.get(data.name, data.name))
                with open(filename, 'w+') as file:
                    file.write(data.body)

    names = []
    req = tensord_pb2.Requests()
    req.model = model
    req.version = version
    for dir_name in os.listdir(input_dir):
        dir = os.path.join(input_dir, dir_name)
        if not os.path.isdir(dir):
            continue

        print "request: ", dir_name, "..."
        names.append(dir_name)
        request = req.request.add()
        for file_name in os.listdir(dir):
            file = os.path.join(dir, file_name)
            if os.path.isdir(file):
                continue
            data = request.data.add()
            data.name = input_alias.get(file_name, file_name)
            with open(file, 'r') as _file:
                data.body = _file.read()

        if len(names) >= batch_size:
            predict(names, req, metrics)

            names = []
            req = tensord_pb2.Requests()
            req.model = model
            req.version = version

    if len(names) > 0:
        predict(names, req, metrics)

    print 'predict_batch: -----'
    print 'model: {}, version: {}'.format(model, version)
    print 'response time: {:.2f}ms / count: {} = {:.2f}ms'.format(
        metrics[0] / 1000, metrics[1], metrics[0] * 1000 / metrics[1])

    channel.close()


def main(args):
    if args['predict']:
        predict(args)
    elif args['predict_batch']:
        predict_batch(args)


if __name__ == "__main__":
    main(docopt(__doc__, version='tensorc 0.5'))
