import struct
import numpy as np


def pack_feature_into_stream(feature, big_endian=True, dtype='f'):
    fmt = '>' if big_endian else '<'
    fmt += str(len(feature)) + dtype
    # print "pack format:", fmt
    stream = struct.pack(fmt, *feature)

    return stream


def unpack_feature_from_stream(stream, big_endian=True, dtype='f'):
    feat_len = len(stream) / 4
    fmt = '>' if big_endian else '<'
    fmt += str(feat_len) + dtype
    # print "unpack format:", fmt

    feature = np.array(struct.unpack(fmt, stream))

    return feat_len, feature
