proto_file = 'python/aisdk/proto/eval_pb2.py'
lines = open(proto_file).read().split('\n')
lines[1] = '# pylint: skip-file\n' + lines[1]
open(proto_file, 'w').write('\n'.join(lines))
